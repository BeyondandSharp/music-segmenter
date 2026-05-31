import io
import os
import glob
import logging
import shutil
import regex
import time
import requests
import asyncio
import functools
from shazamio import Shazam

from utils.logging import save_timestamps


semaphore = asyncio.Semaphore(3)
myshazam = Shazam()


async def shazam_orig(file, **kwargs):
    """Legacy single-shot recognition (uses shazamio's own segment_duration)."""
    match = await _shazam_bytes_or_path(file)
    return shazam_title(match), match


async def shazam_multi_orig(
    file,
    positions=(0.10, 0.50, 0.90),
    sample_length_s=12,
    **kwargs,
):
    """Try recognition at multiple positions; return first successful match."""
    from pydub import AudioSegment as _AudioSegment
    audio = _AudioSegment.from_file(file)
    duration_ms = len(audio)
    sample_ms = int(sample_length_s * 1000)
    last_exc = None
    for pos in positions:
        start_ms = int(duration_ms * pos)
        start_ms = min(start_ms, max(0, duration_ms - sample_ms))
        chunk = audio[start_ms: start_ms + sample_ms]
        buf = io.BytesIO()
        chunk.export(buf, format='mp3')
        try:
            match = await _shazam_bytes_or_path(buf.getvalue())
            logging.info(['shazam_multi_orig matched at', f'{int(pos*100)}%', file])
            return shazam_title(match), match
        except (KeyError, IndexError) as exc:
            last_exc = exc
            logging.debug(['shazam_multi_orig no match at', f'{int(pos*100)}%', file])
    raise IndexError(
        f'shazam_multi_orig: all {len(positions)} offsets failed for {file}'
    ) from last_exc


async def shazaming(
    outdir, media, shazam_coverart_path='',
    shazam_func=None, ignore_fails=False,
    sample_positions=(0.10, 0.50, 0.90),
    sample_length_s=12,
):
    if shazam_func is None:
        shazam_func = functools.partial(
            shazam_multi_orig,
            positions=sample_positions,
            sample_length_s=sample_length_s,
        )
    mediab = os.path.basename(media)
    files = glob.glob(os.path.join(
        outdir, '*' + os.path.splitext(mediab)[0][1:] + '_*'
    ))
    await asyncio.gather(*[shazam_threaded(
        file, shazam_coverart_path=shazam_coverart_path,
        shazam_func=shazam_func, ignore_fails=ignore_fails
    ) for file in files])
    save_timestamps(mediab=mediab,
                    key='shazam', val=[
                        os.path.basename(x)
                        for x in glob.glob(
                            os.path.join(
                                outdir, f"*{mediab[1: mediab.rfind('.')]}*")
                        )
                    ])


async def shazam_threaded(
    file, shazam_coverart_path='',
    shazam_func=shazam_orig, ignore_fails=True
):
    results = {}
    # Skip files that have already been processed by Shazam.
    # After Shazam renames, the filename has "_ARTIST-TITLE" appended after the
    # time-range part (e.g. "_S.S.SS~E.E.EE_ARTIST-TITLE"). We detect this by
    # checking whether a '_' appears after the '~' end-time separator.
    fn_base = os.path.splitext(os.path.basename(file))[0]
    if '~' in fn_base:
        after_end_time = fn_base[fn_base.rfind('~') + 1:]
        if '_' in after_end_time:
            return  # already shazamed
    elif ' by ' in file:  # legacy filename format fallback
        return
    filename = file[:file.rfind('.')]
    fileext = file[len(filename):]
    fn = os.path.basename(filename)
    logging.info(['shazaming', fn])
    try:
        # match = shazam(file, stop_at_first_match = 1)[-1]
        # results[fn] = shazam_title(match)
        results[fn], match = await shazam_func(file)
        try:
            logging.info([fn, 'shazam found to be', results[fn]])
        except UnicodeEncodeError:
            logging.warning(
                [fn, 'shazam found but cant show unicode burr durr'])
        # results[fn] = [title, artist]  (from shazam_title)
        artist = results[fn][1]
        title = results[fn][0]
        renamed_file = os.path.join(
            os.path.dirname(file),
            (fn + f'_{artist}-{title}_shazam') + fileext
        )
        shutil.move(file, renamed_file)
        if os.path.isdir(shazam_coverart_path):
            shazam_coverart(match, renamed_file, shazam_coverart_path)
    except (IndexError, KeyError):
        logging.error([fn, 'shazam failed'])
    except Exception:
        if not ignore_fails:
            raise


async def _shazam_bytes_or_path(data):
    """Low-level: call shazamio.recognize with a file path or bytes."""
    async with semaphore:
        match = await myshazam.recognize(data)
        time.sleep(3)
        return match['track']


async def shazam(mp3):
    """Backwards-compat wrapper: recognise a file path directly."""
    return await _shazam_bytes_or_path(mp3)


def legalize_filename(fn):
    if regex.search(r'\p{IsHangul}', fn) is not None:
        raise KoreanCharException(fn)
    illegal_list = [
        [':', ' '],
        ['"', ''],
        [r'/', ''],
        [r'?', ''],
        [r'*', ''],
        ['\'', ''],
        ['<', ''],
        ['>', ''],
    ]
    for i in illegal_list:
        fn = fn.replace(i[0], i[1])
    return fn


def shazam_title(match):
    title = legalize_filename(match['title'])
    if 'in the style of' in title.lower():
        artist = title[
            title.lower().index('in the style of ') +
            len('in the style of '):]
        artist = artist[:artist.index(')')]
    else:
        artist = legalize_filename(match['subtitle'])

    return [
        legalize_filename(match['title']),
        legalize_filename(match['subtitle']),
    ]


def shazam_coverart(match, fn, outdir):
    try:
        albumart = match['images']['coverarthq']
        req = requests.get(albumart)
        with open(os.path.join(
            outdir, os.path.basename(fn) + albumart[albumart.rfind('.'):]
        ), 'wb') as f:
            f.write(req.content)
    except Exception:
        pass


class KoreanCharException(BaseException):
    pass
