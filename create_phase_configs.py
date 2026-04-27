def create_phase_configs(number_of_samples: int):
    """
    Generate phase configs for Piper TTS training based on dataset size.

    Valid range: 10 – 10000 samples

    Encodes:
    - Sublinear epoch scaling
    - Stronger scaling for core + consolidation
    - Stable early phases
    - Gradual batch size increase
    - Stable LR ratios (critical for GAN balance)
    """

    # --- Clamp input ---
    n = max(10, min(10000, number_of_samples))

    # --- Normalize relative to baseline (720 samples) ---
    k = n / 720.0

    # Sublinear scaling factor (main control law)
    scale = 1.0 + 0.5 * (k - 1.0)

    # Soft cap for very large datasets
    if n > 3000:
        scale *= 0.9
    if n > 6000:
        scale *= 0.85

    # --- Batch scaling (slow, stabilizing) ---
    def scale_batch(base, max_val):
        val = int(round(base * (1.0 + 0.25 * (k - 1.0))))
        return min(max_val, max(base, val))

    # --- Epoch scaling helpers ---
    def scale_epochs(base, strength=1.0):
        return int(round(base * (1.0 + strength * (scale - 1.0))))

    # --- Phase configs ---
    PHASE_CONFIGS = {

        "warmup": {
            # Never scale warmup
            "MAX_EPOCHS": 10,
            "BATCH_SIZE": 8,
            "LEARNING_RATE": 5e-5,
            "LEARNING_RATE_D": 2.0e-5,
            "LR_DECAY": 0.9999,
            "LR_DECAY_D": 0.9999,
        },

        "alignment": {
            # Slight scaling only
            "MAX_EPOCHS": max(15, min(35, scale_epochs(20, 0.3))),
            "BATCH_SIZE": 8,
            "LEARNING_RATE": 7e-5,
            "LEARNING_RATE_D": 2.8e-5,
            "LR_DECAY": 0.9999,
            "LR_DECAY_D": 0.9999,
        },

        "core_training": {
            # Main scaling phase
            "MAX_EPOCHS": max(70, min(180, scale_epochs(90, 1.0))),
            "BATCH_SIZE": scale_batch(12, 28),
            "LEARNING_RATE": 9e-5,
            "LEARNING_RATE_D": 3.5e-5,
            "LR_DECAY": 0.9999,
            "LR_DECAY_D": 0.9999,
        },

        "consolidation": {
            # Strong scaling but slightly less than core
            "MAX_EPOCHS": max(60, min(160, scale_epochs(80, 0.9))),
            "BATCH_SIZE": scale_batch(18, 32),
            "LEARNING_RATE": 6e-5,
            "LEARNING_RATE_D": 2.2e-5,
            "LR_DECAY": 0.99993,
            "LR_DECAY_D": 0.99993,
        },

        "fine_tuning": {
            # Light scaling (diminishing returns)
            "MAX_EPOCHS": max(50, min(120, scale_epochs(60, 0.6))),
            "BATCH_SIZE": scale_batch(22, 36),
            "LEARNING_RATE": 4e-5,
            "LEARNING_RATE_D": 1.5e-5,
            "LR_DECAY": 0.99994,
            "LR_DECAY_D": 0.99994,
        },
    }

    return PHASE_CONFIGS
