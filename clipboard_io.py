"""
剪贴板读写。

从剪贴板读取图片 / 向剪贴板写入文本。
"""

from __future__ import annotations

import time

import pyperclip
from PIL import Image, ImageGrab

from exceptions import ClipboardError


def read_image() -> Image.Image | None:
    """从剪贴板读取图片。

    Returns:
        剪贴板为 PIL Image -> 直接返回
        剪贴板为文件路径列表 -> 取第一个路径，Image.open 读取
        剪贴板为空 / 纯文本 / 其他 -> 返回 None
    """
    data = ImageGrab.grabclipboard()

    if isinstance(data, Image.Image):
        return data

    if isinstance(data, list):
        if data:
            try:
                return Image.open(data[0])
            except Exception:
                return None

    return None


def write_text(text: str) -> None:
    """将文本写入剪贴板。

    失败时重试 3 次（间隔 0.5s），仍失败抛 ClipboardError。

    Raises:
        ClipboardError: 写入失败。
    """
    for attempt in range(3):
        try:
            pyperclip.copy(text)
            return
        except Exception:
            if attempt < 2:
                time.sleep(0.5)
            else:
                raise ClipboardError("写入剪贴板失败，已重试 3 次")
