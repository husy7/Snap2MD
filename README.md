# Snap2MD

基于 [Ollama](https://ollama.com/) + GLM 视觉模型的本地 OCR 工具：截图后按热键，识别剪贴板中的图片并把识别结果（Markdown 格式）写回剪贴板。无需联网、无需调用云端 API，全程在本地运行。

## 功能特性

- **一键识别**：全局热键 `Win+Shift+Ctrl` 触发，识别剪贴板中的图片
- **Markdown 输出**：表格自动转换为 Markdown 表格（支持 `colspan`），数学公式、代码块按 Markdown 语法输出
- **本地推理**：通过 Ollama 调用本地 GLM OCR 模型，零外发流量
- **常驻显存**：启动时后台线程预热模型并设置 `keep_alive=-1`，消除冷启动延迟，不阻塞主流程
- **确定性采样**：`temperature=0.0` + `top_p=0.1`，识别结果稳定可复现
- **重复抑制**：多层末尾重复检测（指纹 / 段落 / 正则 / 行块四重），截断模型生成结束后的死循环填充
- **后处理工具**：reasoning 泄漏剥离、HTML 表格残片清理（已实现并通过测试，默认未接线，可按需启用）
- **Windows 通知**：识别完成 / 出错时弹出 Toast 提示
- **跨次去重**：与上次结果一致时跳过写入，避免冗余
- **非阻塞防抖**：`threading.Lock` 非阻塞获取，连续按键自动丢弃

## 工作流程

```
Win+Shift+Ctrl
   │
   ▼
clipboard_io.read_image()      ── 从剪贴板读取图片（PIL Image 或文件路径）
   │
   ▼
ocr_engine.recognize(img)      ── 压缩 → base64 → POST /api/chat → 提取 message.content → 去重尾 → 表格转 MD
   │
   ▼
clipboard_io.write_text(raw)   ── 识别结果写回剪贴板（失败重试 3 次）
   │
   ▼
notifier.success(...)          ── 终端日志 + Windows Toast
```

程序启动时后台执行 `ocr_engine.warmup()`（`/api/generate` 空请求）预加载模型并常驻显存，失败静默不阻塞启动。

## 项目结构

```
glm_ocr_loacl_use/
├── main.py            # 入口 + 主流程编排（防抖锁、跨次去重、后台预热）
├── config.py          # 纯常量配置（Ollama / Prompt / 图片 / 热键 / 通知）
├── clipboard_io.py    # 剪贴板读写（读图 + 写文本重试 3 次）
├── ocr_engine.py      # Ollama API 调用（压缩 + base64 + POST /api/chat + 表格转换 + 重复截断 + 后处理工具）
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
- **模型**：`glm-ocr`（可按需在 `config.py` 中更换其他多模态模型）

## 安装

```bash
# 1. 克隆仓库

# 2. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate     # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 拉取 OCR 模型（如未拉取）
ollama pull glm-ocr
```

## 使用

1. 确认 Ollama 服务已启动：`ollama serve`
2. 运行主程序：

   ```bash
   python main.py
   ```

3. 用截图工具（Win+Shift+S 等）截取屏幕，图片自动进入剪贴板
4. 按 `Win+Shift+Ctrl` 触发识别，结果会自动写入剪贴板，可直接 `Ctrl+V` 粘贴

> **提示**：`keyboard` 库注册全局热键在 Windows 上需要管理员权限，建议以管理员身份运行终端。

## 配置

所有配置集中在 `config.py`，按需修改后重启程序生效：

| 常量 | 默认值 | 说明 |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `MODEL` | `glm-ocr` | OCR 模型名 |
| `REQUEST_TIMEOUT` | `180` | 单次请求超时（秒） |
| `NUM_CTX` | `8192` | 上下文窗口，OCR 无需 128K（KV cache 显存大幅缩减） |
| `KEEP_ALIVE` | `-1` | 模型永久驻留显存，消除冷启动 |
| `TEMPERATURE` | `0.0` | 采样温度，低值保证输出确定性 |
| `TOP_P` | `0.1` | 核采样阈值，配合低温度抑制发散 |
| `REPEAT_PENALTY` | `1.2` | >1.0 抑制重复，温和档不破坏正常代码 |
| `NUM_PREDICT` | `512` | 输出上限兜底，防止失控重复撑满 |
| `PROMPT` | `你将对图片{b64}进行ocr。请输出识别结果。` | 识别提示词 |
| `MAX_IMAGE_SIDE` | `2000` | 图片最大边长（超过等比缩放） |
| `JPEG_QUALITY` | `85` | 压缩质量 |
| `HOTKEY` | `Win+Shift+ctrl` | 全局触发热键 |
| `TOAST_TITLE` | `glm-OCR:md格式` | Toast 通知标题 |

## 测试

```bash
pytest -v
```

覆盖：

- `ocr_engine`：识别主流程、图片压缩（大图缩放 / RGBA 转 RGB）、`/api/chat` payload（`options.num_ctx` / `repeat_penalty` / `num_predict` / `keep_alive`）、warmup、末尾重复截断、reasoning 泄漏剥离、HTML 表格残片清理
- `clipboard_io`：剪贴板读写
- 集成边界：纯白 / 超大 / 极小图片、空剪贴板、并发触发防抖

## 异常处理

`exceptions.py` 定义统一异常层次，所有异常均继承自 `AppError`：

- `ClipboardError` —— 剪贴板读写失败
- `OllamaConnectionError` —— 无法连接 Ollama
- `OllamaTimeoutError` —— 请求超时
- `OllamaAPIError` —— API 返回非 200 或响应字段缺失

主流程在 `pipeline()` 中捕获 `AppError`，通过 `notifier.error` 输出，不向上抛出，保证热键循环不被中断。

## 许可证

MIT License。
