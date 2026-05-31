#  Load the API
import os
import glob
import logging
import tempfile
import asyncio

from segment.shazam import shazaming
from segment.segment import extract_mah_stuff, extract_music, segment_wrapper,\
    SEGMENT_THRES, TimestampMismatch
from utils.config import load as load_config


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='ina music segment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--media', type=str, required=True, help='local video/audio file path')
    parser.add_argument(
        '--outdir', type=str, default=tempfile.gettempdir(),
        help='directory extracted segments will be saved into')
    parser.add_argument(
        '--config', type=str, default=None,
        help='path to a YAML config file (defaults to config.yaml in the working directory)')

    # ── output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        '--soundonly', action='store_true', default=None,
        help='extract audio only (MP3); omit to extract video segments')
    parser.add_argument(
        '--cleanup', action='store_true', default=None,
        help='delete the original media file once completed.')

    # ── segmenter ─────────────────────────────────────────────────────────────
    parser.add_argument(
        '--max_segment_length', type=int, default=None,
        help='max seconds per inaSpeechSegmenter pass (avoids RAM overflow). '
             f'Default from config (built-in: {SEGMENT_THRES}s)')
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='inaSpeechSegmenter batch size (config default: 32)')
    parser.add_argument(
        '--energy_ratio', type=float, default=None,
        help='inaSpeechSegmenter energy ratio (config default: 0.03)')

    # ── extraction ────────────────────────────────────────────────────────────
    parser.add_argument(
        '--seg_connect', type=int, default=None,
        help='max gap (s) between music segments to merge them (config default: 5)')
    parser.add_argument(
        '--seg_thres', type=int, default=None,
        help='minimum music segment duration for initial filter, seconds (config default: 60)')
    parser.add_argument(
        '--seg_thres_final', type=int, default=None,
        help='minimum music segment duration for final filter, seconds (config default: 80)')

    # ── shazam ────────────────────────────────────────────────────────────────
    parser.add_argument(
        '--shazam', action='store_true', default=None,
        help='shazam the extracted files to identify songs')
    parser.add_argument(
        '--shazam_coverart', type=str, default=None,
        help='directory to save Shazam cover art images')
    parser.add_argument(
        '--shazam_multithread', type=int, default=1,
        help='use multithreaded shazam')

    # ── acrcloud ──────────────────────────────────────────────────────────────
    parser.add_argument(
        '--acrcloud', action='store_true', default=None,
        help='use ACRCloud to identify extracted segments')
    parser.add_argument(
        '--acrcloud_key', type=str, default=None,
        help='ACRCloud access key (overrides config)')
    parser.add_argument(
        '--acrcloud_secret', type=str, default=None,
        help='ACRCloud access secret (overrides config)')
    parser.add_argument(
        '--acrcloud_host', type=str, default=None,
        help='ACRCloud host (overrides config)')

    # ── audd ───────────────────────────────────────────────────────────────────
    parser.add_argument(
        '--audd', action='store_true', default=None,
        help='use AudD to identify extracted segments')
    parser.add_argument(
        '--audd_token', type=str, default=None,
        help='AudD API token (overrides config and AUDD_API_TOKEN env var)')

    # ── acoustid ──────────────────────────────────────────────────────────────
    parser.add_argument(
        '--acoustid', action='store_true', default=None,
        help='use AcoustID (chromaprint) to identify extracted segments')
    parser.add_argument(
        '--acoustid_key', type=str, default=None,
        help='AcoustID API key (overrides config; register free at acoustid.org)')

    # ── yamnet ────────────────────────────────────────────────────────────────
    parser.add_argument(
        '--yamnet', action='store_true', default=None,
        help='use YAMNet to classify each segment as "singing" or "music" '
             'and append the label to the output filename')
    parser.add_argument(
        '--yamnet_threshold', type=float, default=None,
        help='combined singing-class score threshold for YAMNet (config default: 0.10)')

    # ── boundary refinement ───────────────────────────────────────────────────
    parser.add_argument(
        '--boundary', action='store_true', default=None,
        help='enable VAD-based boundary refinement (Silero VAD or pyannote.audio)')
    parser.add_argument(
        '--boundary_engine', type=str, default=None, choices=['silero', 'pyannote'],
        help='VAD engine to use for boundary refinement (config default: silero)')
    parser.add_argument(
        '--boundary_window', type=float, default=None,
        help='search window around each boundary in seconds (config default: 8)')
    parser.add_argument(
        '--boundary_min_silence', type=float, default=None,
        help='minimum silence duration to snap to, seconds (config default: 0.3)')
    parser.add_argument(
        '--pyannote_token', type=str, default=None,
        help='HuggingFace token required by pyannote.audio')

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)-8s %(message)s',
        handlers=[
            logging.FileHandler('./inaseg_inst.log'),
            logging.StreamHandler()
        ])
    args = parser.parse_args()

    # ── Build config overrides from non-None CLI args ─────────────────────────
    cli_overrides = {}
    if args.max_segment_length is not None or args.batch_size is not None \
            or args.energy_ratio is not None:
        cli_overrides['segmenter'] = {}
        if args.max_segment_length is not None:
            cli_overrides['segmenter']['max_segment_length'] = args.max_segment_length
        if args.batch_size is not None:
            cli_overrides['segmenter']['batch_size'] = args.batch_size
        if args.energy_ratio is not None:
            cli_overrides['segmenter']['energy_ratio'] = args.energy_ratio

    if args.seg_connect is not None or args.seg_thres is not None \
            or args.seg_thres_final is not None:
        cli_overrides['extraction'] = {}
        if args.seg_connect is not None:
            cli_overrides['extraction']['seg_connect'] = args.seg_connect
        if args.seg_thres is not None:
            cli_overrides['extraction']['seg_thres'] = args.seg_thres
        if args.seg_thres_final is not None:
            cli_overrides['extraction']['seg_thres_final'] = args.seg_thres_final

    if args.yamnet is not None or args.yamnet_threshold is not None:
        cli_overrides['yamnet'] = {}
        if args.yamnet:
            cli_overrides['yamnet']['enabled'] = True
        if args.yamnet_threshold is not None:
            cli_overrides['yamnet']['singing_threshold'] = args.yamnet_threshold

    if args.boundary is not None or args.boundary_engine is not None \
            or args.boundary_window is not None or args.boundary_min_silence is not None \
            or args.pyannote_token is not None:
        cli_overrides['boundary'] = {}
        if args.boundary:
            cli_overrides['boundary']['enabled'] = True
        if args.boundary_engine is not None:
            cli_overrides['boundary']['engine'] = args.boundary_engine
        if args.boundary_window is not None:
            cli_overrides['boundary']['search_window'] = args.boundary_window
        if args.boundary_min_silence is not None:
            cli_overrides['boundary']['min_silence_duration'] = args.boundary_min_silence
        if args.pyannote_token is not None:
            cli_overrides['boundary']['pyannote_token'] = args.pyannote_token

    if args.soundonly is not None or args.cleanup is not None:
        cli_overrides['output'] = {}
        if args.soundonly:
            cli_overrides['output']['soundonly'] = True
        if args.cleanup:
            cli_overrides['output']['cleanup'] = True

    if args.shazam is not None or args.shazam_coverart is not None:
        cli_overrides['shazam'] = {}
        if args.shazam:
            cli_overrides['shazam']['enabled'] = True
        if args.shazam_coverart is not None:
            cli_overrides['shazam']['coverart_dir'] = args.shazam_coverart

    if args.acrcloud is not None or args.acrcloud_key is not None \
            or args.acrcloud_secret is not None or args.acrcloud_host is not None:
        cli_overrides['acrcloud'] = {}
        if args.acrcloud:
            cli_overrides['acrcloud']['enabled'] = True
        if args.acrcloud_key is not None:
            cli_overrides['acrcloud']['access_key'] = args.acrcloud_key
        if args.acrcloud_secret is not None:
            cli_overrides['acrcloud']['access_secret'] = args.acrcloud_secret
        if args.acrcloud_host is not None:
            cli_overrides['acrcloud']['host'] = args.acrcloud_host

    if args.audd is not None or args.audd_token is not None:
        cli_overrides['audd'] = {}
        if args.audd:
            cli_overrides['audd']['enabled'] = True
        if args.audd_token is not None:
            cli_overrides['audd']['api_token'] = args.audd_token

    if args.acoustid is not None or args.acoustid_key is not None:
        cli_overrides['acoustid'] = {}
        if args.acoustid:
            cli_overrides['acoustid']['enabled'] = True
        if args.acoustid_key is not None:
            cli_overrides['acoustid']['api_key'] = args.acoustid_key

    cfg = load_config(config_path=args.config, overrides=cli_overrides or None)
    logging.info(f'Config: segmenter={vars(cfg.segmenter)} '
                 f'extraction={vars(cfg.extraction)} '
                 f'boundary.enabled={cfg.boundary.enabled}')

    media = args.media
    if not os.path.isfile(media):
        raise FileNotFoundError(f'Media file not found: {media}')

    outdir = args.outdir
    if len(glob.glob(os.path.join(
            outdir,
            f'*{os.path.splitext(os.path.basename(media))[0][1:]}_*'))) == 0:
        # Required for ROCm < 7.13 to activate the DXG/WSL GPU path.
        # setdefault keeps any value the caller already exported.
        os.environ.setdefault('HSA_ENABLE_DXG_DETECTION', '1')
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        logging.info(f'TF visible GPUs: {gpus}')
        for _gpu in gpus:
            tf.config.experimental.set_memory_growth(_gpu, True)
        tf.get_logger().setLevel(logging.WARNING)
        try:
            timestamps = []
            saved_timestamp = extract_music(
                segment_wrapper(media, cfg=cfg),
                cfg=cfg)

            # ── boundary refinement ───────────────────────────────────────────
            if cfg.boundary.enabled and saved_timestamp:
                from segment.boundary import refine_timestamps
                logging.info(
                    f'Refining boundaries with {cfg.boundary.engine} VAD '
                    f'(window=±{cfg.boundary.search_window}s) ...')
                saved_timestamp = refine_timestamps(media, saved_timestamp, cfg)

            # ── YAMNet classification ─────────────────────────────────────────
            yamnet_labels = None
            if cfg.yamnet.enabled and saved_timestamp:
                from segment.yamnet import classify_segments
                logging.info('Running YAMNet classification on extracted segments...')
                yamnet_labels = classify_segments(
                    media, saved_timestamp,
                    singing_threshold=cfg.yamnet.singing_threshold)
                logging.info(['YAMNet labels', list(zip(saved_timestamp, yamnet_labels))])

            extract_mah_stuff(
                media, segmented_stamps=saved_timestamp,
                outdir=outdir, rev=False,
                timestamps=timestamps,
                soundonly=cfg.output.soundonly,
                yamnet_labels=yamnet_labels)
            saved_timestamp = None
        except TimestampMismatch:
            raise
    else:
        logging.warning((
            'segmentation', media, 'stopped to prevent possible duplication'))
    logging.info(['segmentation', media, 'successful'])
    if cfg.output.cleanup and os.path.isfile(media):
        os.remove(media)
    # ── 按优先级运行识别引擎 ───────────────────────────────────────────────────
    def _count_unrecognized():
        """统计 outdir 中属于 media 且尚未被识别的切片文件数。"""
        mediab = os.path.basename(media)
        stem = os.path.splitext(mediab)[0][1:]
        count = 0
        for f in glob.glob(os.path.join(outdir, '*' + stem + '_*')):
            fn_base = os.path.splitext(os.path.basename(f))[0]
            if '~' in fn_base:
                after = fn_base[fn_base.rfind('~') + 1:]
                if '_' in after:
                    continue  # 已识别
            count += 1
        return count

    def _run_shazam():
        asyncio.run(shazaming(
            outdir, media, cfg.shazam.coverart_dir,
            sample_positions=cfg.shazam.sample_positions,
            sample_length_s=cfg.shazam.sample_length))

    def _run_acrcloud():
        from segment.acrcloud import acrcloudding
        acrcloudding(outdir, media, cfg)

    def _run_audd():
        from segment.audd_recog import auddding
        auddding(outdir, media, cfg)

    def _run_acoustid():
        from segment.acoustid_recog import acoustidding
        acoustidding(outdir, media, cfg)

    recognizers = []
    if cfg.shazam.enabled:
        recognizers.append((cfg.shazam.priority, 'shazam', _run_shazam))
    if cfg.acrcloud.enabled:
        recognizers.append((cfg.acrcloud.priority, 'acrcloud', _run_acrcloud))
    if cfg.audd.enabled:
        recognizers.append((cfg.audd.priority, 'audd', _run_audd))
    if cfg.acoustid.enabled:
        recognizers.append((cfg.acoustid.priority, 'acoustid', _run_acoustid))

    recognizers.sort(key=lambda x: x[0], reverse=True)
    for pri, name, runner in recognizers:
        if _count_unrecognized() == 0:
            logging.info('所有切片均已识别，跳过剩余识别引擎。')
            break
        logging.info(f'运行识别引擎: {name} (priority={pri})')
        runner()

    import sys
    sys.exit(0)
