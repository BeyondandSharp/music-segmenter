"""
Boundary refinement for music segments.

After inaSpeechSegmenter returns coarse [start, end] timestamps, this module
searches a small window around each boundary for a quiet region and snaps the
cut point to the nearest silence, producing cleaner edits.

Supported engines
-----------------
silero   : Silero VAD (https://github.com/snakers4/silero-vad)
           Requires: torch, torchaudio
pyannote : pyannote.audio voice-activity-detection pipeline
           Requires: pyannote.audio, a HuggingFace token, and acceptance of
           the model license at https://huggingface.co/pyannote/voice-activity-detection
"""

import logging
import subprocess
from types import SimpleNamespace

import numpy as np

from utils.timestamp import timestamp2sec, sec2timestamp

# ── Audio extraction helper ───────────────────────────────────────────────────

def _extract_pcm(media: str, start_sec: float, end_sec: float,
                 sample_rate: int = 16000) -> np.ndarray:
    """Extract a short window of audio as float32 PCM via ffmpeg."""
    cmd = [
        'ffmpeg',
        '-ss', str(start_sec),
        '-to', str(end_sec),
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
            f'ffmpeg failed: {result.stderr.decode(errors="replace")}'
        )
    return np.frombuffer(result.stdout, dtype=np.float32)


# ── Silence gap detector (engine-agnostic) ────────────────────────────────────

def _find_best_cut(activity_intervals, window_duration: float,
                   target_offset: float, min_silence: float,
                   align: str) -> float:
    """
    Given a list of (start, end) *active* intervals (in seconds, relative to
    window start), find the best cut point closest to *target_offset*.

    The "best cut point" is the midpoint of the longest silence gap whose
    duration >= *min_silence* and that is closest to *target_offset*.

    Parameters
    ----------
    activity_intervals : list of (float, float)
        Active (speech/music) segments relative to window start, in seconds.
    window_duration : float
        Total window duration in seconds.
    target_offset : float
        The original boundary offset relative to window start (seconds).
    min_silence : float
        Minimum silence duration to be considered a valid cut point.
    align : str
        'nearest' → midpoint of nearest silence gap
        'start_of_silence' → beginning of nearest silence gap

    Returns
    -------
    float
        Best cut offset relative to window start (seconds), or *target_offset*
        if no suitable silence is found.
    """
    # Build silence gaps from active intervals
    silence_gaps = []
    prev_end = 0.0
    for seg_start, seg_end in sorted(activity_intervals):
        if seg_start - prev_end >= min_silence:
            silence_gaps.append((prev_end, seg_start))
        prev_end = max(prev_end, seg_end)
    if window_duration - prev_end >= min_silence:
        silence_gaps.append((prev_end, window_duration))

    if not silence_gaps:
        logging.debug('boundary: no silence gap found; keeping original boundary')
        return target_offset

    # Choose the gap whose midpoint is closest to target_offset
    def gap_point(gap):
        if align == 'start_of_silence':
            return gap[0]
        return (gap[0] + gap[1]) / 2.0

    best = min(silence_gaps, key=lambda g: abs(gap_point(g) - target_offset))
    return gap_point(best)


# ── Silero VAD engine ─────────────────────────────────────────────────────────

_silero_model = None
_silero_utils = None


def _load_silero(repo: str = 'snakers4/silero-vad'):
    global _silero_model, _silero_utils
    if _silero_model is None:
        import torch
        logging.info('Boundary: loading Silero VAD...')
        _silero_model, _silero_utils = torch.hub.load(
            repo_or_dir=repo,
            model='silero_vad',
            force_reload=False,
            onnx=False,
        )
    return _silero_model, _silero_utils


def _silero_activity(waveform: np.ndarray, sample_rate: int,
                     repo: str) -> list:
    """Return list of (start_sec, end_sec) active segments via Silero VAD."""
    import torch
    model, utils = _load_silero(repo)
    get_speech_timestamps = utils[0]
    tensor = torch.from_numpy(waveform)
    raw = get_speech_timestamps(
        tensor, model,
        sampling_rate=sample_rate,
        min_speech_duration_ms=100,
        min_silence_duration_ms=int(300),
        return_seconds=True,
    )
    return [(seg['start'], seg['end']) for seg in raw]


# ── pyannote.audio engine ─────────────────────────────────────────────────────

_pyannote_pipeline = None


def _load_pyannote(token: str):
    global _pyannote_pipeline
    if _pyannote_pipeline is None:
        from pyannote.audio import Pipeline
        logging.info('Boundary: loading pyannote VAD pipeline...')
        _pyannote_pipeline = Pipeline.from_pretrained(
            'pyannote/voice-activity-detection',
            use_auth_token=token or None,
        )
    return _pyannote_pipeline


def _pyannote_activity(waveform: np.ndarray, sample_rate: int,
                       token: str) -> list:
    """Return list of (start_sec, end_sec) active segments via pyannote."""
    import io
    import torch
    import torchaudio
    pipeline = _load_pyannote(token)
    tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, samples)
    audio_dict = {'waveform': tensor, 'sample_rate': sample_rate}
    vad_result = pipeline(audio_dict)
    return [(seg.start, seg.end) for seg in vad_result.get_timeline().support()]


# ── Public API ────────────────────────────────────────────────────────────────

def refine_boundary(media: str, time_sec: float, is_start: bool,
                    cfg: SimpleNamespace) -> float:
    """
    Refine a single boundary using the configured VAD engine.

    Parameters
    ----------
    media : str
        Path to the source media file.
    time_sec : float
        Original boundary position in seconds from file start.
    is_start : bool
        True if refining a segment *start* boundary, False for *end*.
    cfg : SimpleNamespace
        ``cfg.boundary`` namespace from the loaded config.

    Returns
    -------
    float
        Refined boundary position in seconds.
    """
    sw = cfg.search_window
    win_start = max(0.0, time_sec - sw)
    win_end = time_sec + sw

    try:
        waveform = _extract_pcm(media, win_start, win_end)
    except Exception:
        logging.exception(f'Boundary: audio extraction failed around {time_sec:.1f}s')
        return time_sec

    if waveform.size == 0:
        return time_sec

    target_offset = time_sec - win_start  # relative position inside window
    window_duration = win_end - win_start

    try:
        if cfg.engine == 'pyannote':
            activity = _pyannote_activity(waveform, 16000, cfg.pyannote_token)
        else:  # default: silero
            activity = _silero_activity(waveform, 16000, cfg.silero_repo)
    except Exception:
        logging.exception(
            f'Boundary ({cfg.engine}): VAD failed around {time_sec:.1f}s; '
            'keeping original boundary'
        )
        return time_sec

    cut_offset = _find_best_cut(
        activity_intervals=activity,
        window_duration=window_duration,
        target_offset=target_offset,
        min_silence=cfg.min_silence_duration,
        align=cfg.align,
    )

    refined = win_start + cut_offset
    delta = refined - time_sec
    logging.info(
        f'Boundary [{("start" if is_start else "end")}] '
        f'{sec2timestamp(time_sec)} → {sec2timestamp(refined)} '
        f'(Δ{delta:+.2f}s, engine={cfg.engine})'
    )
    return refined


def refine_timestamps(media: str, timestamps: list,
                      cfg: SimpleNamespace) -> list:
    """
    Refine a list of ``[start_hms, end_hms]`` timestamp pairs.

    Parameters
    ----------
    media : str
        Path to the source media file.
    timestamps : list of [str, str]
        Each element is ``[start_hms, end_hms]`` in ``HH:MM:SS`` format.
    cfg : SimpleNamespace
        Full config namespace; uses ``cfg.boundary``.

    Returns
    -------
    list of [str, str]
        Refined timestamp pairs in the same ``HH:MM:SS`` format.
    """
    bcfg = cfg.boundary
    result = []
    for start_hms, end_hms in timestamps:
        start_sec = timestamp2sec(start_hms)
        end_sec = timestamp2sec(end_hms)

        new_start = refine_boundary(media, start_sec, is_start=True, cfg=bcfg)
        new_end = refine_boundary(media, end_sec, is_start=False, cfg=bcfg)

        # Sanity: don't let refinement invert or collapse a segment
        if new_end - new_start < 5:
            logging.warning(
                f'Boundary refinement collapsed segment '
                f'{start_hms}-{end_hms}; reverting to original'
            )
            result.append([start_hms, end_hms])
        else:
            result.append([sec2timestamp(new_start), sec2timestamp(new_end)])
    return result
