# MOSS-TTS First-Class E2E 脚本说明

## 脚本位置
`tools/tts/moss-tts-firstclass-e2e.py`

## 功能
该脚本把以下链路封装为一次命令执行：

1. 用 `moss-tts-build-generation-ref.py` 构建 `generation.input.bin`
2. 调用 `llama-moss-tts` 进行 first-class backbone 生成 raw audio codes
3. 用 `moss-tts-audio-decode.py` + ONNX audio tokenizer 解码为 WAV

输入：`text`（可选 `reference audio`）
输出：`wav`

中间产物（`generation.input.bin`、`raw.codes.bin`）会写入临时目录并在结束后自动删除。

## 必需参数
- `--model-gguf`：MOSS-TTS first-class GGUF 模型
- `--tokenizer-dir`：包含 `tokenizer.json` 的目录
- `--onnx-encoder`：MOSS Audio Tokenizer encoder ONNX
- `--onnx-decoder`：MOSS Audio Tokenizer decoder ONNX
- `--output-wav`：输出 wav 路径
- `--text` 或 `--text-file`：二选一

## 常用可选参数
- `--reference-audio`：参考音频（24kHz）
- `--text-temperature`：默认 `1.5`
- `--audio-temperature`：默认 `1.7`
- `--max-new-tokens`：默认 `512`
- `--n-gpu-layers`：默认读取 `MOSS_TTS_N_GPU_LAYERS`，未设置时默认 `1`
- `--python-bin`：指定 Python 解释器
- `--audio-decoder-cpu`：强制 ONNX 解码走 CPU
- `--cpu-audio-encode`：参考音频编码走 CPU
- `--build`：运行前自动构建 `llama-moss-tts`

## `tokenizer-dir` 是什么
`tokenizer-dir` 不是 ONNX 目录，它是文本 tokenizer 目录，至少要有：

- `tokenizer.json`

通常来自 Qwen3 backbone tokenizer 的提取目录。例如：
`weights/extracted/qwen3_backbone`

## 示例
### 1) text + reference 音色克隆
```bash
python tools/tts/moss-tts-firstclass-e2e.py \
  --model-gguf /path/to/moss_delay_firstclass_f16.gguf \
  --tokenizer-dir /path/to/weights/extracted/qwen3_backbone \
  --onnx-encoder /path/to/MOSS-Audio-Tokenizer-ONNX/encoder.onnx \
  --onnx-decoder /path/to/MOSS-Audio-Tokenizer-ONNX/decoder.onnx \
  --text-file /path/to/text.txt \
  --reference-audio /path/to/reference_24k.wav \
  --output-wav /path/to/output.wav
```

### 2) 不带 reference
```bash
python tools/tts/moss-tts-firstclass-e2e.py \
  --model-gguf /path/to/moss_delay_firstclass_f16.gguf \
  --tokenizer-dir /path/to/weights/extracted/qwen3_backbone \
  --onnx-encoder /path/to/MOSS-Audio-Tokenizer-ONNX/encoder.onnx \
  --onnx-decoder /path/to/MOSS-Audio-Tokenizer-ONNX/decoder.onnx \
  --text "清晨的青藏高原，空气稀薄而寒冷。" \
  --output-wav /path/to/output.wav
```

## 输出
脚本结束时会打印：

- `wav` 路径
- `wav_info`（采样率、声道、帧数、时长）

注：`llama-moss-tts` 在该链路中不再做 generation parity 返回码判定；只要流程成功会返回 0。
