import os
import yaml


SAVE_YAML_PATH = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    'save.yaml')


def _load_yaml(path: str, default: dict = {}) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or default
    except FileNotFoundError:
        return default


def _save_yaml(path: str, data: dict) -> None:
    try:
        os.replace(path, path + '.old')
    except FileNotFoundError:
        pass
    with open(path, 'w') as f:
        yaml.dump(data, f)


def save_timestamps(mediab: str, key: object, val: object, config: str = SAVE_YAML_PATH) -> None:
    save = _load_yaml(config, {})
    if not mediab in save:
        save[mediab] = {}
    save[mediab][key] = val
    _save_yaml(config, save)
    return save
