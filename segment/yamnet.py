"""
YAMNet-based audio classifier.
Classifies extracted music segments as 'singing' (vocal-led) or 'music' (instrumental).

Requires: tensorflow-hub
Model: https://tfhub.dev/google/yamnet/1  (521 AudioSet classes)
"""

import csv
import logging
import subprocess

import numpy as np

# Class name substrings that indicate singing / vocal activity
SINGING_CLASS_KEYWORDS = [
    'singing',
    'choir',
    'vocal music',
    'a capella',
    'opera',
    'chant',
    'humming',
    'yodeling',
    'mantra',
    'child singing',
    'synthetic singing',
]

DEFAULT_SINGING_THRESHOLD = 0.10  # combined singing-class score to call a segment 'singing'

_yamnet_model = None
_class_names = None


def load_yamnet():
    """Lazily load YAMNet from TF Hub (cached after first call)."""
    global _yamnet_model, _class_names
    if _yamnet_model is None:
        import os
        import tensorflow as tf
        import tensorflow_hub as hub
        # Enable ROCm/DXG GPU path (no-op on ROCm >= 7.13, safe to set always).
        os.environ.setdefault('HSA_ENABLE_DXG_DETECTION', '1')
        # Configure GPU memory growth.  Raises RuntimeError if TF is already
        # initialised (e.g. inaSpeechSegmenter ran first) — that is fine.
        try:
            for _gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(_gpu, True)
        except RuntimeError:
            pass
        logging.info('YAMNet: loading model from TF Hub ...')
        _yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        class_map_path = _yamnet_model.class_map_path().numpy().decode('utf-8')
        with open(class_map_path, newline='', encoding='utf-8') as f:
            _class_names = [row['display_name'] for row in csv.DictReader(f)]
        singing_class_names = [
            name for name in _class_names
            if any(kw in name.lower() for kw in SINGING_CLASS_KEYWORDS)
        ]
        logging.info(f'YAMNet: loaded. Singing-related classes: {singing_class_names}')
    return _yamnet_model, _class_names


def _extract_audio(media, start_hms=None, end_hms=None, sample_rate=16000):
    """
    Extract an audio window from *media* as raw float32 PCM via ffmpeg.
    start_hms / end_hms are HH:MM:SS strings (same format used elsewhere in the project).
    """
    cmd = ['ffmpeg']
    if start_hms:
        cmd += ['-ss', start_hms]
    if end_hms:
        cmd += ['-to', end_hms]
    cmd += [
        '-i', media,
        '-ar', str(sample_rate),
        '-ac', '1',
        '-f', 'f32le',
        '-loglevel', 'error',
        'pipe:1',
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f'ffmpeg audio extraction failed: {result.stderr.decode(errors="replace")}'
        )
    return np.frombuffer(result.stdout, dtype=np.float32)


def classify_segment(media, start_hms, end_hms,
                     singing_threshold=DEFAULT_SINGING_THRESHOLD):
    """
    Classify one media segment as ``'singing'`` or ``'music'`` using YAMNet.

    A segment is labelled *singing* when the sum of mean per-frame scores for
    all singing-related AudioSet classes exceeds *singing_threshold*.

    Parameters
    ----------
    media : str
        Path to the source media file (anything ffmpeg can read).
    start_hms : str
        Segment start time in ``HH:MM:SS`` format.
    end_hms : str
        Segment end time in ``HH:MM:SS`` format.
    singing_threshold : float
        Combined singing-class score threshold (default 0.10).

    Returns
    -------
    str
        ``'singing'`` or ``'music'``
    """
    import tensorflow as tf

    model, class_names = load_yamnet()

    waveform = _extract_audio(media, start_hms, end_hms)
    if waveform.size == 0:
        logging.warning(
            f'YAMNet: empty audio for {start_hms}-{end_hms}, defaulting to music'
        )
        return 'music'

    scores, _, _ = model(tf.constant(waveform, dtype=tf.float32))
    mean_scores = scores.numpy().mean(axis=0)  # (521,)

    singing_score = float(sum(
        mean_scores[i]
        for i, name in enumerate(class_names)
        if any(kw in name.lower() for kw in SINGING_CLASS_KEYWORDS)
    ))

    top_idx = mean_scores.argsort()[-5:][::-1]
    top5 = [(class_names[i], round(float(mean_scores[i]), 4)) for i in top_idx]
    logging.info(
        f'YAMNet [{start_hms} - {end_hms}]  singing_score={singing_score:.4f}  '
        f'top5={top5}'
    )

    return 'singing' if singing_score >= singing_threshold else 'music'


def classify_segments(media, timestamps,
                      singing_threshold=DEFAULT_SINGING_THRESHOLD):
    """
    Classify a list of ``[start_hms, end_hms]`` timestamp pairs.

    Parameters
    ----------
    media : str
        Path to the source media file.
    timestamps : list of [str, str]
        Each element is ``[start_hms, end_hms]`` in ``HH:MM:SS`` format.
    singing_threshold : float
        Passed through to :func:`classify_segment`.

    Returns
    -------
    list of str
        Parallel list of labels: ``'singing'`` or ``'music'``.
    """
    labels = []
    for start_hms, end_hms in timestamps:
        try:
            label = classify_segment(media, start_hms, end_hms, singing_threshold)
        except Exception:
            logging.exception(
                f'YAMNet: classification failed for {start_hms}-{end_hms}, '
                f'defaulting to music'
            )
            label = 'music'
        labels.append(label)
    return labels
