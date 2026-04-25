#!/usr/bin/env python3
"""Minimal checkpoint downloader for rhasspy/piper-checkpoints.

Usage:
  python checkpoint.py --list
  python checkpoint.py --download 5 /path/to/dest
  python checkpoint.py --download en/en_US/ljspeech/medium/ljspeech-1000.ckpt
  python checkpoint.py --download 5 /path/to/dest --save_as mymodel

The tool lists only checkpoint files (*.ckpt) and downloads the selected file to
the destination directory (or current working directory by default).
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DATASET_API = "https://huggingface.co/api/datasets/rhasspy/piper-checkpoints"
DATASET_BASE_URL = "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main"


def fetch_checkpoint_index():
    request = urllib.request.Request(
        DATASET_API,
        headers={"User-Agent": "Piper-Tools/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Failed to load checkpoint index: {exc}")

    siblings = data.get("siblings")
    if not isinstance(siblings, list):
        raise SystemExit("Unexpected dataset index format from Hugging Face.")

    checkpoints = [
        sibling["rfilename"]
        for sibling in siblings
        if sibling.get("rfilename", "").endswith(".ckpt")
    ]
    if not checkpoints:
        raise SystemExit("No checkpoint files found in the remote dataset.")
    return sorted(checkpoints)


def list_checkpoints(checkpoints):
    print("Available checkpoints from rhasspy/piper-checkpoints:")
    print("Use '--download <number>' or '--download <remote-path>' to fetch a checkpoint.")
    for idx, checkpoint in enumerate(checkpoints, start=1):
        print(f"  {idx}. {checkpoint}")


def resolve_selection(selection, checkpoints):
    if selection.isdigit():
        index = int(selection)
        if 1 <= index <= len(checkpoints):
            resolved = checkpoints[index - 1]
            print(f"Selected checkpoint #{index}: {resolved}")
            return resolved
        raise SystemExit(
            f"Checkpoint index {index} is out of range (1-{len(checkpoints)})."
        )

    if selection in checkpoints:
        return selection

    matches = [chk for chk in checkpoints if chk.endswith(selection)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(
            f"Selection '{selection}' is ambiguous. Use --list and choose a number."
        )

    raise SystemExit(
        f"Checkpoint '{selection}' was not found. Use --list to see available checkpoints."
    )


def get_remote_file_info(url: str):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Piper-Tools/1.0"},
        method="HEAD",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        total = int(response.getheader("Content-Length") or 0)
        accept_ranges = response.getheader("Accept-Ranges", "none").lower()
        return total, accept_ranges


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_file = dest.with_name(dest.name + ".part")
    existing_temp_size = temp_file.stat().st_size if temp_file.exists() else 0

    if dest.exists() and dest.stat().st_size > 0:
        print(f"Skipping existing file: {dest}")
        return

    try:
        total, accept_ranges = get_remote_file_info(url)
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Failed to inspect remote file: {exc}")

    if existing_temp_size and total and existing_temp_size >= total:
        temp_file.rename(dest)
        print(f"Renamed completed partial file to {dest}")
        return

    request_headers = {"User-Agent": "Piper-Tools/1.0"}
    if existing_temp_size and accept_ranges != "none":
        request_headers["Range"] = f"bytes={existing_temp_size}-"

    try:
        request = urllib.request.Request(url, headers=request_headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            status = getattr(response, "status", None) or response.getcode()
            if existing_temp_size and status == 200:
                print("Server does not support resume. Restarting download from the beginning.")
                existing_temp_size = 0
                mode = "wb"
            else:
                mode = "ab" if existing_temp_size else "wb"

            remaining = int(response.getheader("Content-Length") or 0)
            downloaded = existing_temp_size
            target_total = total or (existing_temp_size + remaining)
            chunk_size = 8192

            with open(temp_file, mode) as out_file:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if target_total:
                        pct = downloaded / target_total
                        bar_len = 40
                        filled = int(pct * bar_len)
                        bar = "#" * filled + "-" * (bar_len - filled)
                        print(
                            f"\rDownloading {dest.name}: [{bar}] {downloaded}/{target_total} bytes",
                            end="",
                            flush=True,
                        )
                if target_total:
                    print()

        temp_file.rename(dest)
    except KeyboardInterrupt:
        print(f"\nDownload interrupted by user. Partial file saved as: {temp_file}")
        raise SystemExit(1)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise SystemExit(f"Download failed: {exc}")


def download_checkpoint(checkpoint: str, destination: Path, save_as: str | None = None) -> Path:
    url = f"{DATASET_BASE_URL}/{checkpoint}?download=true"
    dest_name = Path(save_as).name if save_as else Path(checkpoint).name
    if dest_name.lower().endswith(".ckpt"):
        dest_name = dest_name
    else:
        dest_name = f"{dest_name}.ckpt"
    dest_file = destination / dest_name
    print(f"Downloading checkpoint to: {dest_file}")
    download_file(url, dest_file)
    return dest_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="List or download Piper checkpoints from rhasspy/piper-checkpoints."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list",
        action="store_true",
        help="List available checkpoint files with numbers.",
    )
    group.add_argument(
        "--download",
        metavar="SELECTOR",
        help="Download a checkpoint by number or remote path.",
    )
    parser.add_argument(
        "--save_as",
        metavar="NAME",
        help="Save the downloaded checkpoint under NAME.ckpt in the destination directory.",
    )
    parser.add_argument(
        "destination",
        nargs="?",
        help="Destination folder for download (default: current working directory).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list:
        checkpoints = fetch_checkpoint_index()
        list_checkpoints(checkpoints)
        return

    checkpoints = fetch_checkpoint_index()
    selection = resolve_selection(args.download, checkpoints)
    destination = Path(args.destination or Path.cwd()).expanduser()
    checkpoint_path = download_checkpoint(selection, destination, args.save_as)
    print(f"Downloaded: {checkpoint_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully.")
        raise SystemExit(1)
