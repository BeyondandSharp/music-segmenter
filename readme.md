# SongExtractor

从视频/音频文件中自动分割并识别音乐片段。

## 功能

1. **分割**：用 [inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter) 检测音乐段落并切片
2. **识别**：按优先级依次调用多个识别引擎，成功后重命名文件（含引擎后缀）
   - ACRCloud（priority=60）
   - Shazam（priority=50）
   - AudD（priority=40）
   - AcoustID / chromaprint（priority=30）
3. **可选**：YAMNet 人声/纯音乐分类、Silero/pyannote 边界精化

输出文件名格式：`原名_NN_SS.SS~EE.EE_ARTIST-TITLE_引擎名.mp3`

## 依赖

- Python 3.10+
- `ffmpeg`（含 chromaprint 支持）及 `ffprobe` 在系统 PATH 中

```
pip install -r requirements.txt
```

## 用法

```bash
python inaseg.py --media /path/to/video.mp4 --outdir ./output
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--media` | 输入文件路径（必填） |
| `--outdir` | 切片输出目录（默认系统临时目录） |
| `--config` | 自定义配置文件路径（默认 `config.yaml`） |
| `--soundonly` | 仅导出 MP3，不导出视频流 |
| `--cleanup` | 完成后删除原始文件 |
| `--shazam` | 启用 Shazam 识别 |
| `--acrcloud` | 启用 ACRCloud 识别 |
| `--audd` | 启用 AudD 识别 |
| `--acoustid` | 启用 AcoustID 识别 |

完整参数见 `python inaseg.py --help`。

## 配置

编辑 `config.yaml` 填写各引擎的 API Key，并按需调整分割阈值、采样位置等参数。

```yaml
acrcloud:
  enabled: true
  access_key: "..."
  access_secret: "..."

shazam:
  enabled: true

audd:
  enabled: true
  api_token: "..."

acoustid:
  enabled: true
  api_key: "..."
```
