"""
AudD 音频识别模块
依赖：pip install audd pydub
文档：https://docs.audd.io/sdks/python
"""

import glob
import io
import logging
import os
import shutil

from segment.shazam import legalize_filename, KoreanCharException
from utils.logging import save_timestamps

# audd.recognize() 每段须为 5~25 秒
_AUDD_MIN_S = 5
_AUDD_MAX_S = 25


def _recognize_file(client, filepath: str, positions: list, sample_length: int) -> tuple:
    """
    在文件的多个相对位置各取 sample_length 秒进行识别（用 pydub 切片后传 bytes），
    返回首个成功匹配的 (title, artist)；全部失败抛 KeyError。
    """
    from pydub import AudioSegment as _AudioSegment

    sample_length = max(_AUDD_MIN_S, min(sample_length, _AUDD_MAX_S))
    sample_ms = sample_length * 1000

    audio = _AudioSegment.from_file(filepath)
    duration_ms = len(audio)

    last_exc = None
    for pos in positions:
        start_ms = int(duration_ms * pos)
        start_ms = min(start_ms, max(0, duration_ms - sample_ms))
        label = f'{int(pos * 100)}%({start_ms // 1000}s)'

        chunk = audio[start_ms: start_ms + sample_ms]
        buf = io.BytesIO()
        chunk.export(buf, format='mp3')

        try:
            result = client.recognize(buf.getvalue())
            if result is None:
                logging.debug(['AudD no match at', label, os.path.basename(filepath)])
                last_exc = KeyError(f'no match at {label}')
                continue
            title = legalize_filename(result.title or '')
            artist = legalize_filename(result.artist or '')
            logging.info(['AudD matched at', label, os.path.basename(filepath)])
            return title, artist
        except KoreanCharException:
            raise
        except Exception as exc:
            logging.debug(['AudD error at', label, str(exc)])
            last_exc = exc

    raise KeyError(
        f'AudD: all {len(positions)} offsets failed for {os.path.basename(filepath)}'
    ) from last_exc


def auddding(outdir: str, media: str, cfg) -> None:
    """
    对 outdir 里属于 media 的所有切片文件运行 AudD 识别，
    识别成功后重命名为 原名_ARTIST-TITLE.ext，用法与 shazaming() / acrcloudding() 对称。
    """
    from audd import AudD, AudDAuthenticationError, AudDQuotaError, AudDAPIError

    audd_cfg = cfg.audd
    token = audd_cfg.api_token or 'test'
    if not audd_cfg.api_token:
        logging.warning('AudD: api_token 未配置，使用 "test" token（10次/天）')

    positions = list(audd_cfg.sample_positions)
    sample_length = int(audd_cfg.sample_length)

    mediab = os.path.basename(media)
    files = glob.glob(os.path.join(
        outdir, '*' + os.path.splitext(mediab)[0][1:] + '_*'
    ))

    if not files:
        logging.warning(['AudD: no segment files found in', outdir])
        return

    with AudD(token) as client:
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
            logging.info(['AudD recognizing', fn])

            try:
                title, artist = _recognize_file(client, file, positions, sample_length)
                renamed_file = os.path.join(
                    os.path.dirname(file),
                    fn + f'_{artist}-{title}_audd' + fileext,
                )
                shutil.move(file, renamed_file)
                logging.info(['AudD renamed to', os.path.basename(renamed_file)])
            except KoreanCharException:
                logging.error(['AudD: Korean chars in filename, skip', fn])
            except AudDAuthenticationError as exc:
                logging.error(['AudD: authentication error, aborting', str(exc)])
                break
            except AudDQuotaError as exc:
                logging.error(['AudD: quota exhausted, aborting', str(exc)])
                break
            except AudDAPIError as exc:
                logging.error(['AudD API error for', fn, str(exc)])
            except KeyError:
                logging.warning(['AudD: no match for', fn])
            except Exception:
                logging.exception(['AudD unexpected error for', fn])

    save_timestamps(
        mediab=mediab,
        key='audd',
        val=[
            os.path.basename(x)
            for x in glob.glob(os.path.join(outdir, f"*{mediab[1: mediab.rfind('.')]}*"))
        ],
    )
