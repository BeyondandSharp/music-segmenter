"""
ACRCloud 音频识别模块
依赖：pip install pyacrcloud pydub
"""

import glob
import json
import logging
import os
import shutil

from utils.logging import save_timestamps

# 重用 shazam 模块里的文件名合法化逻辑
from segment.shazam import legalize_filename, KoreanCharException


def _acrcloud_title(data: dict) -> list[str]:
    """从 ACRCloud 响应中提取 [title, artist]，失败抛 KeyError。"""
    status = data.get('status', {})
    if status.get('code') != 0:
        raise KeyError(f"ACRCloud status {status.get('code')}: {status.get('msg')}")
    music_list = data.get('metadata', {}).get('music', [])
    if not music_list:
        raise KeyError('ACRCloud: no music metadata in response')
    m = music_list[0]
    title = legalize_filename(m.get('title', ''))
    artists = ', '.join(a.get('name', '') for a in m.get('artists', []))
    artist = legalize_filename(artists)
    return [title, artist]


def _recognize_file(recognizer, filepath: str, positions: list[float], sample_length: int) -> dict:
    """
    在文件的多个相对位置各取 sample_length 秒进行识别，
    返回首个成功的原始 ACRCloud dict；全部失败则抛 KeyError。
    """
    try:
        duration_ms = int(recognizer.get_duration_ms_by_file(filepath))
        duration_s = duration_ms / 1000
    except Exception as exc:
        raise KeyError(f'ACRCloud: cannot get duration for {filepath}') from exc

    last_exc = None
    for pos in positions:
        start_s = int(duration_s * pos)
        # 确保末尾留有足够时长
        start_s = min(start_s, max(0, int(duration_s) - sample_length))
        label = f'{int(pos * 100)}%({start_s}s)'
        try:
            raw = recognizer.recognize_by_file(filepath, start_s, sample_length)
            data = json.loads(raw)
            if data.get('status', {}).get('code') == 0 and data.get('metadata', {}).get('music'):
                logging.info(['ACRCloud matched at', label, os.path.basename(filepath)])
                return data
            logging.debug(['ACRCloud no match at', label, os.path.basename(filepath),
                           data.get('status', {}).get('msg')])
            last_exc = KeyError(f'no match at {label}')
        except Exception as exc:
            logging.debug(['ACRCloud error at', label, str(exc)])
            last_exc = exc

    raise KeyError(
        f'ACRCloud: all {len(positions)} offsets failed for {os.path.basename(filepath)}'
    ) from last_exc


def acrcloudding(outdir: str, media: str, cfg) -> None:
    """
    对 outdir 里属于 media 的所有切片文件运行 ACRCloud 识别，
    识别成功后重命名为 原名_ARTIST-TITLE.ext，用法与 shazaming() 对称。
    """
    from acrcloud.recognizer import ACRCloudRecognizer

    acr_cfg = cfg.acrcloud
    if not acr_cfg.access_key or not acr_cfg.access_secret:
        logging.error('ACRCloud: access_key / access_secret 未配置，跳过识别')
        return

    recognizer = ACRCloudRecognizer({
        'host': acr_cfg.host,
        'access_key': acr_cfg.access_key,
        'access_secret': acr_cfg.access_secret,
        'timeout': acr_cfg.timeout,
    })

    positions = list(acr_cfg.sample_positions)
    sample_length = int(acr_cfg.sample_length)

    mediab = os.path.basename(media)
    files = glob.glob(os.path.join(
        outdir, '*' + os.path.splitext(mediab)[0][1:] + '_*'
    ))

    if not files:
        logging.warning(['ACRCloud: no segment files found in', outdir])
        return

    for file in sorted(files):
        fn_base = os.path.splitext(os.path.basename(file))[0]
        # 跳过已经识别过的文件（~ 后面有 _）
        if '~' in fn_base:
            after_end_time = fn_base[fn_base.rfind('~') + 1:]
            if '_' in after_end_time:
                continue

        filename = file[:file.rfind('.')]
        fileext = file[len(filename):]
        fn = os.path.basename(filename)
        logging.info(['ACRCloud recognizing', fn])

        try:
            data = _recognize_file(recognizer, file, positions, sample_length)
            title_artist = _acrcloud_title(data)
            artist = title_artist[1]
            title = title_artist[0]
            renamed_file = os.path.join(
                os.path.dirname(file),
                fn + f'_{artist}-{title}_acrcloud' + fileext,
            )
            shutil.move(file, renamed_file)
            logging.info(['ACRCloud renamed to', os.path.basename(renamed_file)])
        except (KeyError, KoreanCharException):
            logging.error(['ACRCloud failed for', fn])
        except Exception:
            logging.exception(['ACRCloud unexpected error for', fn])

    save_timestamps(
        mediab=mediab,
        key='acrcloud',
        val=[
            os.path.basename(x)
            for x in glob.glob(os.path.join(outdir, f"*{mediab[1: mediab.rfind('.')]}*"))
        ],
    )
