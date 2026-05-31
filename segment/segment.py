#  Load the API
from inaSpeechSegmenter import Segmenter  # noqa: E402
import os
from threading import Thread
import gc
import logging
import tensorflow as tf

from utils.ffmpeg import get_segment_process_length_array, ffmpeg
from utils.timestamp import fix_missing_stamps, sec2timestamp
from utils.logging import save_timestamps

# 媒体流最大时长处理（秒）；1G内存的进程推荐用10分钟/600秒，16G可以支持5小时，6GB VRAM可以支持5小时左右。
SEGMENT_THRES = 800
# 识歌分段最小阈值（秒），调大了会漏 调小了会多杂谈
EXTRACT_SEG_THRES = 60
# 最终识歌分段最小阈值（秒），调大了漏TV size 调小了多杂谈
EXTRACT_SEG_THRES_FINAL = 80
# 识歌分段连接的阈值（秒），调大了会两首歌分不开 调小了会碎
EXTRACT_SEG_CONNECT = 5
# 大了会碎 小了会两首歌分不开
ENERGY_RATIO = 0.03
# 8GB VRAM 推荐 256
BATCH_SIZE = 32


class TimestampMismatch(Exception):
    pass


def _hms_filesafe(hms: str) -> str:
    """Replace ':' with '.' so the timestamp is safe in Windows filenames."""
    return hms.replace(':', '.')


# select a media to analyse
#  any media supported by ffmpeg may be used (video, audio, urls)


def segment(
        media: str, batch_size: int = BATCH_SIZE, energy_ratio: float = ENERGY_RATIO,
        start_sec: int = None, stop_sec: int = None,
        cfg=None):
    bs = cfg.segmenter.batch_size if cfg else batch_size
    er = cfg.segmenter.energy_ratio if cfg else energy_ratio
    segmenter = Segmenter(
        vad_engine='sm',
        detect_gender=False,
        energy_ratio=er,
        batch_size=bs)
    segmentation = segmenter(media, start_sec=start_sec, stop_sec=stop_sec)
    return segmentation


def segment_wrapper(
        media: str, batch_size: int = BATCH_SIZE,
        energy_ratio: float = ENERGY_RATIO, segment_length_thres: int = 0,
        cfg=None):
    ''''''
    thres = cfg.segmenter.max_segment_length if cfg else segment_length_thres
    result = []
    for i in get_segment_process_length_array(media, thres):
        logging.info([
            'segmenting', media, 'from',
            sec2timestamp(i[0]), 'to', sec2timestamp(i[1])])
        result += segment(
            media, batch_size, energy_ratio,
            start_sec=i[0], stop_sec=i[1], cfg=cfg)
        gc.collect()
        tf.keras.backend.clear_session()
    return result


def extract_music(
        segmentation, segment_thres=EXTRACT_SEG_THRES,
        segment_thres_final=EXTRACT_SEG_THRES_FINAL,
        segment_connect=EXTRACT_SEG_CONNECT, start_padding=1, end_padding=4,
        cfg=None):
    if cfg is not None:
        segment_thres = cfg.extraction.seg_thres
        segment_thres_final = cfg.extraction.seg_thres_final
        segment_connect = cfg.extraction.seg_connect
        start_padding = cfg.extraction.start_padding
        end_padding = cfg.extraction.end_padding
    r = []
    # bridges noEnergy segments that are likely fragmented
    for i in range(len(segmentation)-2, 0, -1):
        if segmentation[i][0] == 'noEnergy' and \
                segmentation[i][2] - segmentation[i][1] < 4 and \
                segmentation[i-1][0] == segmentation[i+1][0]:
            segmentation[i-1] = (
                segmentation[i-1][0],
                segmentation[i-1][1],
                segmentation[i+1][2])
    for i in segmentation:
        if i[0] == 'music' and i[2]-i[1] > segment_thres:
            r.append(['', i[1] - start_padding, i[2] + end_padding])
    for i in range(len(r)-1, 0, -1):
        if r[i][1] - r[i-1][2] < segment_connect:
            r[i-1][2] = r[i][2]
            r[i][1] = r[i][2] + 1
    rf = []
    for i in r:
        if i[1] < 5:
            continue
        if i[2]-i[1] > segment_thres_final:
            rf.append(i)
    return [['{}:{}:{}'.format(str(int(x[1]//3600)),
                               str(int(x[1] % 3600 // 60)),
                               str(int(x[1] % 60)).zfill(2)),
             '{}:{}:{}'.format(str(int(x[2]//3600)),
                               str(int(x[2] % 3600 // 60)),
                               str(int(x[2] % 60)).zfill(2))] for x in rf]


def extract_mah_stuff(
        media, segmented_stamps, outdir=None, rev=False,
        delimited='/', timestamps=[], soundonly=True, yamnet_labels=None):
    nameswitch = False
    timestamps_ext = segmented_stamps
    try:
        if len(timestamps) > 0:
            raise FileNotFoundError()
        for i in open(r"D:\tmp\ytd\timstamp.ini", 'r', encoding='UTF-8'):
            for k in [
                ['\n', ''], ['」', ''], ['~「', ' '], ['「', ' '],
                    ['『', ' '], ['』', ' '], ['　', ' '], ['	', ' '], ['\'', ''],]:
                i = i.replace(k[0], k[1])
            if ' （' in i:
                i = i[:i.find(' （')]
            if ':' in i:
                timestamps.append([i[:i.find(' ')], i[i.find(' ')+1:]])
                # timestamps[-1][1] = timestamps[-1][1][1:]
                if delimited in timestamps[-1][1]:
                    timestamps[-1][1] = [
                        timestamps[-1][1][:timestamps[-1][1].find('/')],
                        timestamps[-1][1][timestamps[-1][1].find('/')+1:]]
                    if rev:
                        timestamps[-1][1].reverse()
                    timestamps[-1][1] = ' by '.join(timestamps[-1][1])
                # timestamps[-1][1].replace('/', ' by ')
                # timestamps[-1][1] = ' by '.join(timestamps[-1][1].split('/'))
                # timestamps[-1][1] = timestamps[-1][1].replace('-', 'by')
                while timestamps[-1][1][0] == ' ':
                    timestamps[-1][1] = timestamps[-1][1][1:]
                while timestamps[-1][1][-1] == ' ':
                    timestamps[-1][1] = timestamps[-1][1][:-1]
        #        nameswitch = True
            elif nameswitch:
                nameswitch = False
                timestamps[-1].append(i)
        if len(timestamps) > 0:  # and len(timestamps) != len(timestamps_ext):
            logging.info('checking timestamp correspondence and removing\
                mismatched ones (come on, are you really gonna do this manually)')
            timestamps = fix_missing_stamps(timestamps, timestamps_ext)
            timestamps_ext = fix_missing_stamps(timestamps_ext, timestamps)
            if len(timestamps) != len(timestamps_ext):
                raise TimestampMismatch(
                    'check timestamp assist', timestamps, timestamps_ext)
        with open(r"D:\tmp\ytd\timstamp.ini", 'w') as f:  # noqa: F841
            pass
    except FileNotFoundError:
        pass
    if len(timestamps) > 0:
        logging.info([
            'timestamp assist',
            [[timestamps[x][0],
              timestamps_ext[x][1], timestamps[x][1], ]
              for x in range(len(timestamps))]])
    else:
        logging.info([
            'extracted timestamps',
            ['{} - {}'.format(x[0], x[1]) for x in timestamps_ext]])
    try:
        for count, x in enumerate(timestamps_ext):
            logging.info(f'{str(count).zfill(2)}: {x[0]} - {x[1]}')
    except Exception:
        pass
    save_timestamps(mediab=os.path.basename(media),
                    key='timestamps', val=timestamps_ext)
    nameswitch = False
    file = media
    filename = file[:file.rfind('.')]
    fileext = file[len(filename):]
    filename = os.path.basename(filename)
    cmds = []
    for i in range(len(timestamps_ext)):
        oud = outdir if outdir else os.path.dirname(file)
        os.makedirs(oud, exist_ok=True)
        encoding = ['-c:v', 'copy', '-c:a', 'copy']  # '-c:v copy -c:a copy'
        if soundonly:
            encoding = ['-vn', '-ab', '320k']  # '-vn -ab 320k'
            fileext = '.mp3'
        yamnet_suffix = f'_{yamnet_labels[i]}' if yamnet_labels and i < len(yamnet_labels) else ''
        try:
            start_hms = timestamps[i][0]
            end_hms = timestamps_ext[i][1]
            time_range = f'{_hms_filesafe(start_hms)}~{_hms_filesafe(end_hms)}'
            cmds.append([
                'ffmpeg',
                '-ss',
                start_hms,
                '-to',
                end_hms,
                '-i',
                file,
                '-reset_timestamps', '1',
            ] + encoding + [
                os.path.join(
                    oud, filename + f'_{str(i).zfill(2)}{yamnet_suffix}_{time_range}' + fileext),
            ] + encoding)
        except Exception:
            start_hms = timestamps_ext[i][0]
            end_hms = timestamps_ext[i][1]
            time_range = f'{_hms_filesafe(start_hms)}~{_hms_filesafe(end_hms)}'
            cmds.append([
                'ffmpeg',
                '-ss',
                start_hms,
                '-to',
                end_hms,
                '-i',
                "{}".format(file),
            ] + encoding + [
                "{}".format(os.path.join(
                    oud, filename + f'_{str(i).zfill(2)}{yamnet_suffix}_{time_range}' + fileext)),
            ])
    k = [Thread(target=ffmpeg, args=(x,)) for x in cmds]
    for i in k:
        i.start()
    for i in k:
        i.join()
    # k[-1].join()
