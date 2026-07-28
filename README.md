# Snap2MD

基于 [Ollama](https://ollama.com/) + GLM 视觉模型的本地 OCR 工具：截图后按热键，识别剪贴板中的图片并把识别结果（Markdown 格式）写回剪贴板。无需联网、无需调用云端 API，全程在本地运行。

## 功能特性

- **一键识别**：全局热键 `Win+Shift+Z` 触发，识别剪贴板中的图片
- **Markdown 输出**：表格自动转换为 Markdown 表格，数学公式、代码块按 Markdown 语法输出
- **本地推理**：通过 Ollama 调用本地 GLM OCR 模型，零外发流量
- **常驻显存**：启动时预热模型并设置 `keep_alive=-1`，消除冷启动延迟
- **重复抑制**：内置末尾重复检测，截断模型生成结束后的死循环填充
- **Windows 通知**：识别完成 / 出错时弹出 Toast 提示
- **跨次去重**：与上次结果一致时跳过写入，避免冗余
- **非阻塞防抖**：`threading.Lock` 非阻塞获取，连续按键自动丢弃

## 工作流程

```
Win+Shift+Z
   │
   ▼
clipboard_io.read_image()      ── 从剪贴板读取图片（PIL Image 或文件路径）
   │
   ▼
ocr_engine.recognize(img)      ── 压缩 → base64 → POST /api/chat → 提取文本 → 去重尾 → 表格转 MD
   │
   ▼
clipboard_io.write_text(raw)   ── 识别结果写回剪贴板（失败重试 3 次）
   │
   ▼
notifier.success(...)          ── 终端日志 + Windows Toast
```

## 项目结构

```
glm_ocr_loacl_use/
├── main.py            # 入口 + 主流程编排（防抖锁、跨次去重）
├── config.py          # 纯常量配置（Ollama / Prompt / 图片 / 热键 / 通知）
├── clipboard_io.py    # 剪贴板读写（读图 + 写文本重试 3 次）
├── ocr_engine.py      # Ollama API 调用（压缩 + base64 + POST + 表格转换 + 重复截断）
├── postprocess.py     # 可选的 re 清理管线（当前未在 pipeline 中调用，保留备用）
├── notifier.py        # 终端日志 + Windows Toast 弹窗
├── hotkey.py          # 全局快捷键注册（keyboard 库）
├── exceptions.py      # AppError 异常层次：ClipboardError / Ollama*Error
├── tests/             # pytest 测试（clipboard_io / ocr_engine / 集成边界）
└── requirements.txt
```

## 环境要求

- **OS**：Windows 10 / 11（依赖 `keyboard` 全局热键与 `plyer` Toast）
- **Python**：3.10+（使用了 `X | None` 联合类型语法）
- **Ollama**：本地已安装并运行，地址默认 `http://localhost:11434`
- **模型**：`glm-ocr:q8_0`（可按需在 `config.py` 中更换其他多模态模型）

## 安装

```bash
# 1. 克隆仓库

# 2. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate     # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 拉取 OCR 模型（如未拉取）
ollama pull glm-ocr:q8_0
```

## 使用

1. 确认 Ollama 服务已启动：`ollama serve`
2. 运行主程序：

   ```bash
   python main.py
   ```

3. 用截图工具（Win+Shift+S 等）截取屏幕，图片自动进入剪贴板
4. 按 `Win+Shift+Z` 触发识别，结果会自动写入剪贴板，可直接 `Ctrl+V` 粘贴

> **提示**：`keyboard` 库注册全局热键在 Windows 上需要管理员权限，建议以管理员身份运行终端。

## 配置

所有配置集中在 `config.py`，按需修改后重启程序生效：

| 常量 | 默认值 | 说明 |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `MODEL` | `glm-ocr:q8_0` | OCR 模型名 |
| `REQUEST_TIMEOUT` | `180` | 单次请求超时（秒） |
| `NUM_CTX` | `8192` | 上下文窗口，OCR 场景无需 128K |
| `KEEP_ALIVE` | `-1` | 模型永久驻留显存，消除冷启动 |
| `REPEAT_PENALTY` | `1.1` | 温和抑制重复，不破坏正常代码 |
| `NUM_PREDICT` | `2048` | 输出上限兜底 |
| `PROMPT` | `强制所有内容使用Markdown语法` | 识别提示词 |
| `MAX_IMAGE_SIDE` | `2000` | 图片最大边长（超过等比缩放） |
| `JPEG_QUALITY` | `85` | 压缩质量 |
| `HOTKEY` | `Win+Shift+Z` | 全局触发热键 |
| `TOAST_TITLE` | `glm-OCR:md格式` | Toast 通知标题 |

## 测试

```bash
pytest -v
```

覆盖 `clipboard_io`、`ocr_engine`（含 API payload、表格转换、末尾重复截断）以及集成边界场景。

## 异常处理

`exceptions.py` 定义统一异常层次，所有异常均继承自 `AppError`：

- `ClipboardError` —— 剪贴板读写失败
- `OllamaConnectionError` —— 无法连接 Ollama
- `OllamaTimeoutError` —— 请求超时
- `OllamaAPIError` —— API 返回非 200 或响应字段缺失

主流程在 `pipeline()` 中捕获 `AppError`，通过 `notifier.error` 输出，不向上抛出，保证热键循环不被中断。

## 许可证

MIT License。详见 [LICENSE](LICENSE)。
