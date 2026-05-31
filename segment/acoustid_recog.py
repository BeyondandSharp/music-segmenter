"""
AcoustID 音频指纹识别模块
通过 FFmpeg chromaprint filter 提取音频指纹（无需 fpcalc），
调用 AcoustID WebService 识别歌曲。

依赖：pip install pyacoustid
      ffmpeg（需含 chromaprint 支持）及 ffprobe 在系统 PATH 中。

指纹提取命令示例：
  ffmpeg -ss 30 -t 30 -i input.mp3 -f chromaprint -fp_format compressed -

文档：https://acoustid.org/webservice
"""

import base64
import glob
import logging
import os
import shutil
import subprocess

from segment.shazam import legalize_filename, KoreanCharException
from utils.logging import save_timestamps

_ACOUSTID_MIN_S = 10


def _get_duration_s(filepath: str) -> float:
    """用 ffprobe 获取音频时长（秒）。"""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        filepath,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f'ffprobe 失败: {r.stderr[:200]}')
    return float(r.stdout.strip())


def _ffmpeg_chromaprint(filepath: str, start_s: float, length_s: int) -> str:
    """
    用 FFmpeg chromaprint filter 提取 compressed 指纹，
    返回 base64url（无 padding）字符串供 AcoustID API 使用。

    等价命令：
      ffmpeg -ss {start_s} -t {length_s} -i {filepath}
             -f chromaprint -fp_format compressed -
    """
    cmd = [
        'ffmpeg',
        '-ss', f'{start_s:.3f}',
        '-t', str(length_s),
        '-i', filepath,
        '-f', 'chromaprint',
        '-fp_format', 'compressed',
        '-',
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    if r.returncode != 0 or not r.stdout:
        stderr = r.stderr.decode(errors='replace')
        if 'chromaprint' in stderr.lower():
            raise RuntimeError(
                'FFmpeg 不支持 chromaprint（缺少 --enable-chromaprint 编译选项），'
                '请下载包含 chromaprint 支持的预编译版本'
            )
        raise RuntimeError(f'ffmpeg chromaprint 失败: {stderr[:300]}')
    # compressed 字节 → base64url 无 padding（AcoustID API 要求的格式）
    return base64.urlsafe_b64encode(r.stdout).rstrip(b'=').decode()


def _parse_lookup(response: dict, min_score: float, filename: str, label: str) -> tuple:
    """从 AcoustID API 响应 dict 中提取 (title, artist)，失败抛 KeyError。"""
    if response.get('status') != 'ok':
        raise KeyError(f'AcoustID API status: {response.get("status")}')
    for hit in response.get('results', []):
        score = float(hit.get('score', 0.0))
        if score < min_score:
            logging.debug(['AcoustID low score', f'{score:.2f}', label, filename])
            continue
        for rec in hit.get('recordings', []):
            title_raw = rec.get('title', '').strip()
            artist_raw = ', '.join(
                a.get('name', '') for a in rec.get('artists', [])
            ).strip()
            if title_raw and artist_raw:
                title = legalize_filename(title_raw)
                artist = legalize_filename(artist_raw)
                logging.info(['AcoustID matched', f'{score:.2f}', label, filename])
                return title, artist
    raise KeyError(f'AcoustID: no usable match at {label} for {filename}')


def _recognize_file(filepath: str, api_key: str, positions: list, sample_length: int,
                    min_score: float) -> tuple:
    """
    在文件的多个相对位置各取 sample_length 秒进行 AcoustID 识别，
    返回首个成功匹配的 (title, artist)；全部失败抛 KeyError。
    """
    import acoustid

    sample_length = max(_ACOUSTID_MIN_S, sample_length)
    duration_s = _get_duration_s(filepath)
    filename = os.path.basename(filepath)

    last_exc = None
    for pos in positions:
        start_s = duration_s * pos
        start_s = min(start_s, max(0.0, duration_s - sample_length))
        label = f'{int(pos * 100)}%({int(start_s)}s)'
        try:
            fp = _ffmpeg_chromaprint(filepath, start_s, sample_length)
            response = acoustid.lookup(api_key, fp, sample_length, meta='recordings')
            return _parse_lookup(response, min_score, filename, label)
        except KoreanCharException:
            raise
        except KeyError as exc:
            last_exc = exc
        except Exception as exc:
            logging.debug(['AcoustID error', label, filename, str(exc)])
            last_exc = exc

    raise KeyError(
        f'AcoustID: all {len(positions)} offsets failed for {filename}'
    ) from last_exc


def acoustidding(outdir: str, media: str, cfg) -> None:
    """
    对 outdir 里属于 media 的所有切片文件运行 AcoustID 识别，
    识别成功后重命名为 原名_ARTIST-TITLE.ext，用法与 shazaming() / acrcloudding() 对称。
    """
    import acoustid

    acfg = cfg.acoustid
    if not acfg.api_key:
        logging.error('AcoustID: api_key 未配置，跳过识别')
        return

    positions = list(acfg.sample_positions)
    sample_length = int(acfg.sample_length)
    min_score = float(acfg.min_score)

    mediab = os.path.basename(media)
    files = glob.glob(os.path.join(
        outdir, '*' + os.path.splitext(mediab)[0][1:] + '_*'
    ))

    if not files:
        logging.warning(['AcoustID: no segment files found in', outdir])
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
        logging.info(['AcoustID recognizing', fn])

        try:
            title, artist = _recognize_file(
                file, acfg.api_key, positions, sample_length, min_score)
            renamed_file = os.path.join(
                os.path.dirname(file),
                fn + f'_{artist}-{title}_acoustid' + fileext,
            )
            shutil.move(file, renamed_file)
            logging.info(['AcoustID renamed to', os.path.basename(renamed_file)])
        except KoreanCharException:
            logging.error(['AcoustID: Korean chars in filename, skip', fn])
        except RuntimeError as exc:
            # FFmpeg 配置问题（不支持 chromaprint 或二进制不在 PATH）
            logging.error(['AcoustID runtime error', str(exc)])
            break
        except acoustid.WebServiceError as exc:
            logging.error(['AcoustID Web Service error for', fn, str(exc)])
        except KeyError:
            logging.warning(['AcoustID: no match for', fn])
        except Exception:
            logging.exception(['AcoustID unexpected error for', fn])

    save_timestamps(
        mediab=mediab,
        key='acoustid',
        val=[
            os.path.basename(x)
            for x in glob.glob(os.path.join(outdir, f"*{mediab[1: mediab.rfind('.')]}*"))
        ],
    )

