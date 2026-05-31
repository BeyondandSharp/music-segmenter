"""
Configuration loader for SongExtractor.

Loads config.yaml and provides a simple dot-accessible namespace.
CLI-supplied values (passed as override_dict) shadow the file values.
"""

import copy
import logging
import os
from types import SimpleNamespace

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# ── Default configuration (mirrors config.yaml) ──────────────────────────────
_DEFAULTS = {
    'segmenter': {
        'batch_size': 32,
        'energy_ratio': 0.03,
        'max_segment_length': 800,
    },
    'extraction': {
        'seg_thres': 60,
        'seg_thres_final': 80,
        'seg_connect': 5,
        'start_padding': 1,
        'end_padding': 4,
    },
    'yamnet': {
        'enabled': False,
        'singing_threshold': 0.10,
        'singing_class_keywords': [
            'singing', 'choir', 'vocal music', 'a capella', 'opera',
            'chant', 'humming', 'yodeling', 'mantra',
            'child singing', 'synthetic singing',
        ],
    },
    'boundary': {
        'enabled': False,
        'engine': 'silero',
        'search_window': 8,
        'min_silence_duration': 0.3,
        'align': 'nearest',
        'pyannote_token': '',
        'silero_repo': 'snakers4/silero-vad',
    },
    'output': {
        'soundonly': False,
        'cleanup': False,
    },
    'shazam': {
        'enabled': False,
        'priority': 50,
        'coverart_dir': '',
        'sample_positions': [0.10, 0.50, 0.90],
        'sample_length': 12,
    },
    'acrcloud': {
        'enabled': False,
        'priority': 60,
        'host': 'identify-ap-southeast-1.acrcloud.com',
        'access_key': '',
        'access_secret': '',
        'timeout': 10,
        'sample_positions': [0.10, 0.50, 0.90],
        'sample_length': 10,
    },
    'audd': {
        'enabled': False,
        'priority': 40,
        'api_token': '',
        'sample_positions': [0.10, 0.50, 0.90],
        'sample_length': 12,
    },
    'acoustid': {
        'enabled': False,
        'priority': 30,
        'api_key': '',

        'sample_positions': [0.10, 0.50, 0.90],
        'sample_length': 30,
        'min_score': 0.70,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        elif value is not None:
            result[key] = value
    return result


def _dict_to_ns(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace."""
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def load(config_path: str = None, overrides: dict = None) -> SimpleNamespace:
    """
    Load configuration.

    Priority (highest → lowest):
      1. *overrides* dict (values from CLI args)
      2. YAML config file at *config_path*
      3. Built-in defaults

    Parameters
    ----------
    config_path : str, optional
        Path to a YAML config file.  If None, looks for ``config.yaml``
        next to this file, then in the current working directory.
    overrides : dict, optional
        Nested dict of values that override the config file.
        Only non-None leaf values are applied.

    Returns
    -------
    SimpleNamespace
        Dot-accessible configuration tree, e.g. ``cfg.extraction.seg_connect``.
    """
    merged = copy.deepcopy(_DEFAULTS)

    # ── locate config file ────────────────────────────────────────────────────
    if config_path is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), '..', 'config.yaml'),
            os.path.join(os.getcwd(), 'config.yaml'),
        ]
        config_path = next((p for p in candidates if os.path.isfile(p)), None)

    if config_path and os.path.isfile(config_path):
        if not _YAML_AVAILABLE:
            logging.warning(
                'pyyaml is not installed; config file will be ignored. '
                'Install it with: pip install pyyaml'
            )
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_cfg = yaml.safe_load(f) or {}
            merged = _deep_merge(merged, file_cfg)
            logging.debug(f'Config loaded from {os.path.abspath(config_path)}')
    else:
        logging.debug('No config.yaml found; using built-in defaults.')

    # ── apply CLI overrides ───────────────────────────────────────────────────
    if overrides:
        merged = _deep_merge(merged, overrides)

    return _dict_to_ns(merged)
