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
    text = _trim_tail_repetition(text)
    return _convert_table(text)


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
                "content": config.PROMPT,
                "images": [b64],
            }
        ],
        "stream": False,
        "options": {
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


def _trim_tail_repetition(text: str) -> str:
    """末尾重复截断：只处理模型生成结束后的死循环填充，不碰正文。

    两级递归检测：
    1. 短模式（2-20 字符）末尾连续重复 4+ 次
    2. 长段（末行块与紧邻前块完全一致）末尾紧邻重复 2+ 次
    """
    # 短模式
    m = re.search(r'([\s\S]{2,20}?)\1{3,}\s*$', text)
    if m:
        return _trim_tail_repetition(text[:m.start()].rstrip())

    # 长段行块重复
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


def warmup() -> None:
    """预热模型：发送空请求让 Ollama 加载模型并常驻显存。"""
    url = f"{config.OLLAMA_URL}/api/generate"
    payload = {
        "model": config.MODEL,
        "keep_alive": config.KEEP_ALIVE,
        "options": {
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
