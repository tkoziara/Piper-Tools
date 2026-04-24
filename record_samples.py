#!/usr/bin/env python3
"""Simple web tool to record/transcribe/collect voice samples.

Usage: python record_samples.py --lang en|pl /path/to/dir

Starts a server on localhost:8765.  Open the page in a browser, record your
voice, edit the recognized sentence and save.  WAV+TXT pair will be created in
specified directory as SampleN.(wav,txt) for English or ProbkaN.* for Polish.

Dependencies (installed by setup_venv.sh):
    flask whisper
and a system installation of ``ffmpeg`` (see install_system_deps.sh).
"""

import argparse
import os
import re
import sys
import tempfile
import subprocess
from pathlib import Path

try:
    import whisper
    # validate that it is the OpenAI whisper package (it provides load_model)
    if not hasattr(whisper, 'load_model'):
        raise ImportError("imported whisper module has no load_model")
except ImportError:
    print("The installed 'whisper' package does not support load_model.")
    print("Make sure to install OpenAI's model: pip install openai-whisper")
    sys.exit(1)

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

args = None
model = None

PREFIX = {"en": "Sample", "pl": "Probka"}

# helper to pick next available index
num_re = re.compile(r"{}(\d+)\.wav".format(re.escape('%s')))

def next_index(dest: Path, prefix: str) -> int:
    """Return next integer index for prefix in dest directory."""
    highest = 0
    for f in dest.iterdir():
        if f.is_file() and f.suffix.lower() == ".wav":
            m = re.match(r"%s(\d+)\.wav" % re.escape(prefix), f.name)
            if m:
                try:
                    v = int(m.group(1))
                    if v > highest:
                        highest = v
                except ValueError:
                    pass
    return highest + 1

@app.route("/", methods=["GET"])
def index():
    # embed simple HTML/JS for recording
    return f"""<!DOCTYPE html>
<html><head><meta charset=utf-8><title>Sample recorder</title></head><body>
<h1>Voice sampler ({args.lang})</h1>
<button id=\"btn\">Start recording</button>
<select id=\"device\"></select><br><br>
<textarea id=\"text\" cols=80 rows=4 placeholder=\"Transcription\"></textarea><br>
<button id=\"save\">Save sample</button>
<audio id=\"player\" controls style=\"display:none;margin-top:10px;\"></audio>
<p id=\"status\"></p>
<script>
let mediaRecorder, audioChunks, lastBlob;
const btn = document.getElementById('btn');
const saveBtn = document.getElementById('save');
const status = document.getElementById('status');
const textArea = document.getElementById('text');
const deviceSel = document.getElementById('device');

async function enumerate() {{
  const devices = await navigator.mediaDevices.enumerateDevices();
  devices.forEach(d=>{{
    if (d.kind === 'audioinput') {{
      const opt = document.createElement('option');
      opt.value=d.deviceId; opt.text=d.label||('mic '+deviceSel.length);
      deviceSel.appendChild(opt);
    }}
  }});
}}
enumerate();

btn.onclick = async ()=>{{
  if (mediaRecorder && mediaRecorder.state === 'recording') {{
    mediaRecorder.stop();
    btn.textContent = 'Start recording';
  }} else {{
    const constraints={{audio:{{deviceId:deviceSel.value?{{exact:deviceSel.value}}:undefined}}}};
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    audioChunks=[];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable=e=>audioChunks.push(e.data);
    mediaRecorder.onstop=()=>{{
      const blob=new Blob(audioChunks);
      lastBlob = blob;
      // show player
      const player = document.getElementById('player');
      player.src = URL.createObjectURL(blob);
      player.style.display = '';
      transcribe(blob);
    }};
    mediaRecorder.start();
    btn.textContent='Stop recording';
  }}
}};

async function transcribe(blob){{
  status.textContent='Transcribing...';
  const form=new FormData();
  form.append('audio', blob, 'sample.webm');
  const resp=await fetch('/transcribe',{{method:'POST',body:form}});
  const data=await resp.json();
  textArea.value=data.text||'';
  status.textContent = 'Done.  Edit text if needed and Save.';
}}

saveBtn.onclick=async ()=>{{
  if (!lastBlob) return;
  status.textContent='Saving...';
  const form=new FormData();
  form.append('audio', lastBlob, 'sample.webm');
  form.append('text', textArea.value);
  await fetch('/save',{{method:'POST',body:form}});
  status.textContent='Saved.';
}};
</script>
</body></html>"""

@app.route("/transcribe", methods=["POST"])
def transcribe():
    f = request.files.get('audio')
    if not f:
        return jsonify(error="no audio"), 400
    tmp = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    f.save(tmp.name)
    # run whisper
    res = model.transcribe(tmp.name, language=args.lang)
    os.unlink(tmp.name)
    return jsonify(text=res.get('text',''))

@app.route("/save", methods=["POST"])
def save():
    f = request.files.get('audio')
    txt = request.form.get('text','')
    if not f:
        return jsonify(error="no audio"), 400
    tmp = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    f.save(tmp.name)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    prefix = PREFIX.get(args.lang,'Sample')
    idx = next_index(dest, prefix)
    wavpath = dest / f"{prefix}{idx}.wav"
    txtpath = dest / f"{prefix}{idx}.txt"
    # convert to wav
    subprocess.run(['ffmpeg','-y','-i',tmp.name,'-ar','22050','-ac','1',str(wavpath)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(txtpath,'w',encoding='utf-8') as o:
        o.write(txt)
    os.unlink(tmp.name)
    return jsonify(success=True)


def main():
    global args, model
    parser = argparse.ArgumentParser(description="Run sample collection server")
    parser.add_argument('--lang', choices=['en','pl'], required=True)
    parser.add_argument('dest', help='directory to save samples')
    args = parser.parse_args()
    print(f"Loading whisper model (this may take a moment)")
    model = whisper.load_model('base')
    port = 8765 if args.lang == 'en' else 8764
    print(f"Model loaded, starting server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    main()
