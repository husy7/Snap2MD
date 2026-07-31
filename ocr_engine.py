"""
Ollama API 调用。

图片 -> base64 -> 调用 Ollama -> 返回原始文本。
"""

from __future__ import annotations

import base64
import io
import re
from html.parser import HTMLParser

import requests
from PIL import Image

import config
from exceptions import (
    OllamaAPIError,
    OllamaConnectionError,
    OllamaTimeoutError,
)


def recognize(img: Image.Image) -> str:
    """PIL Image 进，识别文本出。"""
    b64 = _encode(img)
    resp = _call_api(b64)
    text = _extract(resp)
    text =  _trim_tail_repetition(text)
    return _convert_table(text)


import re

def _trim_tail_repetition(text: str) -> str:
    """检测并移除重复内容，包括循环重复的整段"""
    if not text:
        return text
    
    # 专门检测：如果文本以某段内容开头，且后面重复了该内容
    # 取前 20-50 个字符作为"指纹"
    for length in range(20, 60, 5):
        if len(text) < length * 2:
            break
        fingerprint = text[:length]
        # 如果指纹在后面的文本中出现
        remaining = text[length:]
        if fingerprint in remaining:
            # 找到第一次重复的位置
            repeat_start = remaining.find(fingerprint) + length
            # 截断到重复开始前
            return text[:repeat_start].strip()
    
    # 按段落检测（如果某个段落重复出现）
    paragraphs = [p for p in text.split('\n') if p.strip()]
    if len(paragraphs) > 1:
        first_para = paragraphs[0]
        # 如果第二段和第一段相同或高度相似
        if len(paragraphs) >= 2 and paragraphs[1] == first_para:
            return paragraphs[0]
        # 检查是否有任何段落重复
        seen = set()
        for i, para in enumerate(paragraphs):
            if para in seen:
                # 返回重复之前的所有内容
                return '\n'.join(paragraphs[:i])
            seen.add(para)
    
    # 原有逻辑
    m = re.search(r'([\s\S]{2,20}?)\1{3,}\s*$', text)
    if m:
        return _trim_tail_repetition(text[:m.start()].rstrip())
    
    lines = text.split('\n')
    n = len(lines)
    if n >= 4:
        max_k = min(n // 2, 100)
        for k in range(max_k, 1, -1):
            pattern = '\n'.join(lines[n - k:n])
            if len(pattern) < 21:
                break
            if n - k >= k:
                preceding = '\n'.join(lines[n - 2 * k:n - k])
                if preceding == pattern:
                    return _trim_tail_repetition('\n'.join(lines[:n - k]).rstrip())
    
    return text


def _encode(img: Image.Image) -> str:
    """PIL Image -> 压缩 -> base64 字符串。"""
    w, h = img.size
    if max(w, h) > config.MAX_IMAGE_SIDE:
        scale = config.MAX_IMAGE_SIDE / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=config.JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _call_api(b64: str) -> dict:
    """POST /api/chat，返回原始 JSON dict。"""
    url = f"{config.OLLAMA_URL}/api/chat"
    payload = {
        "model": config.MODEL,
        "messages": [
            {
                "role": "user",
                "images": [b64],
                "content": config.PROMPT,
                
            }
        ],
        "stream": False,
        "options": {
            "temperature": config.TEMPERATURE,
            "top_p": config.TOP_P,
            "num_ctx": config.NUM_CTX,
            "repeat_penalty": config.REPEAT_PENALTY,
            "num_predict": config.NUM_PREDICT,
            
        },
        "keep_alive": config.KEEP_ALIVE,
    }
    try:
        resp = requests.post(url, json=payload, timeout=config.REQUEST_TIMEOUT)
    except requests.exceptions.ConnectionError:
        raise OllamaConnectionError("无法连接到 Ollama，服务在尝试启动")
    except requests.exceptions.Timeout:
        raise OllamaTimeoutError(f"Ollama 请求超时（{config.REQUEST_TIMEOUT}s）")

    if resp.status_code != 200:
        raise OllamaAPIError(f"Ollama 返回非 200 状态码: {resp.status_code}")

    return resp.json()



class _TableParser(HTMLParser):
    """解析 <table> 中的 <tr><td> 结构，提取行数据。"""
    def __init__(self):
        super().__init__()
        self.rows = []
        self._row = None
        self._in_td = False
        self._colspan = 1
        self._buf = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'tr':
            self._row = []
        elif tag in ('td', 'th'):
            self._in_td = True
            self._colspan = int(a.get('colspan', 1))
            self._buf = []

    def handle_data(self, data):
        if self._in_td:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag in ('td', 'th'):
            text = ''.join(self._buf).strip()
            for _ in range(self._colspan):
                self._row.append(text)
            self._in_td = False
        elif tag == 'tr':
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _table_to_markdown(html_block: str) -> str:
    """单个 <table> HTML 块 -> Markdown 表格。"""
    parser = _TableParser()
    parser.feed(html_block)
    if not parser.rows:
        return html_block

    lines = []
    for i, row in enumerate(parser.rows):
        lines.append('| ' + ' | '.join(row) + ' |')
        if i == 0:
            lines.append('| ' + ' | '.join(['---'] * len(row)) + ' |')
    return '\n'.join(lines)


def _convert_table(text: str) -> str:
    """将文本中所有 <table> HTML 表格替换为 Markdown 表格。"""
    def replacer(m):
        return _table_to_markdown(m.group(0))
    return re.sub(r'<table[^>]*>.*?</table>', replacer, text, flags=re.DOTALL)


_REASONING_OPENER_RE = re.compile(
    r'^(?:Actually|Wait|Hmm|Let me|I need to|I\'ll|I should|The prompt|'
    r'Looking at|So the|First|Now I|The image|One detail|I think|'
    r'It seems|The user|Let\'s)\b',
    re.IGNORECASE,
)


def _has_cjk(text: str) -> bool:
    """文本是否包含 CJK 字符（中文/日文/韩文）。"""
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)


def _is_markdown_structure(line: str) -> bool:
    """行是否是 Markdown 结构（标题/表格/代码栅栏/列表）。"""
    s = line.lstrip()
    if not s:
        return False
    if s.startswith('#') or s.startswith('|') or s.startswith('```'):
        return True
    if s.startswith(('- ', '* ', '+ ')):
        return True
    if re.match(r'^\d+\.\s', s):
        return True
    return False


def _strip_reasoning_leakage(text: str) -> str:
    """剥离模型 reasoning 泄漏块。

    glm-ocr 在 prompt 模糊或图片信息少时会把英文思考过程泄漏到 content，
    典型表现：以 "Actually," / "Wait," / "The prompt says" 等开头的多行英文块。

    检测以 reasoning opener 开头、后续为连续英文非结构行的块并整体移除。
    遇到空行 / CJK 字符 / Markdown 结构 / 代码栅栏即停止当前块的剥离。
    代码栅栏（``` ... ```）内的内容一律保留，不参与剥离。

    注意：纯英文 OCR 内容（如英文截图）若以这些词开头会被误删，
    属于已知折衷，必要时调整 _REASONING_OPENER_RE。
    """
    if not text:
        return text

    lines = text.split('\n')
    keep: list[str] = []
    i = 0
    in_code_fence = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 代码栅栏状态跟踪：栅栏内一律保留
        if stripped.startswith('```'):
            in_code_fence = not in_code_fence
            keep.append(line)
            i += 1
            continue

        if (not in_code_fence
                and stripped
                and _REASONING_OPENER_RE.match(stripped)
                and not _has_cjk(stripped)):
            # 进入 reasoning 块，跳过后续连续英文行
            i += 1
            while i < len(lines):
                next_stripped = lines[i].strip()
                if not next_stripped:
                    break  # 空行结束块
                if _has_cjk(next_stripped):
                    break  # 中文内容结束块
                if _is_markdown_structure(next_stripped):
                    break  # Markdown 结构结束块
                i += 1
            # reasoning 行不写入 keep
        else:
            keep.append(line)
            i += 1

    return '\n'.join(keep).strip()


def _strip_incomplete_html_tables(text: str) -> str:
    """移除未闭合的 HTML 表格残片（模型被 NUM_PREDICT 截断时产生）。

    完整的 <table>...</table> 由 _convert_table 转换为 Markdown；
    无 </table> 的残片在此移除，避免污染 _trim_tail_repetition 的末尾检测。

    从末尾向前迭代剥离：每次找到最后一个 <table，若其后无 </table> 则截断。
    """
    while True:
        last_table = text.rfind('<table')
        if last_table == -1:
            return text
        if '</table>' in text[last_table:]:
            return text
        text = text[:last_table].rstrip()




def warmup() -> None:
    """预热模型：发送空请求让 Ollama 加载模型并常驻显存。"""
    url = f"{config.OLLAMA_URL}/api/generate"
    payload = {
        "model": config.MODEL,
        "keep_alive": config.KEEP_ALIVE,
        "options": {
            "temperature": config.TEMPERATURE,
            "top_p": config.TOP_P,
            "num_ctx": config.NUM_CTX,
            "repeat_penalty": config.REPEAT_PENALTY,
            "num_predict": config.NUM_PREDICT,
        },
    }
    try:
        requests.post(url, json=payload, timeout=config.REQUEST_TIMEOUT)
    except requests.RequestException:
        pass


def _extract(response: dict) -> str:
    """从 JSON 中提取 message.content 字符串。"""
    try:
        return response["message"]["content"]
    except (KeyError, TypeError):
        raise OllamaAPIError("Ollama 响应中未找到 message.content")
