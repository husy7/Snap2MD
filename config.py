"""
配置管理。

纯常量模块，其他模块只从此处读取配置。
后续可扩展为读取 .env 或 config.yaml。
"""

# ── Ollama ──────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
MODEL = "glm-ocr"
REQUEST_TIMEOUT = 180  # 秒
NUM_CTX = 8192    # 上下文窗口，OCR 无需 128K，缩小可大幅减少 KV cache 显存
KEEP_ALIVE = -1        # 模型永久驻留显存（-1），消除冷启动
REPEAT_PENALTY = 1.2  # >1.0 抑制重复，1.1 温和档，不破坏正常代码
NUM_PREDICT = 512    # 输出上限兜底，防止失控重复撑满
TEMPERATURE = 0.0
TOP_P = 0.1

# ── Prompt ──────────────────────────────────────────────
PROMPT = '''你将对图片{b64}进行ocr。请输出识别结果。
     '''         # #Recognition,提示词

# ── 图片处理 ────────────────────────────────────────────
MAX_IMAGE_SIDE = 2000  # 超过此尺寸等比缩放
JPEG_QUALITY = 85     # 压缩质量

# ── 快捷键 ──────────────────────────────────────────────
HOTKEY = "Win+Shift+ctrl"

# ── 通知 ──────────────────────────────────────────────
TOAST_TITLE = "glm-OCR:md格式"
