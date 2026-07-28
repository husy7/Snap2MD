"""
全局快捷键监听。

注册全局快捷键，触发回调函数。
"""

from __future__ import annotations

from collections.abc import Callable

import keyboard

import config


def register(callback: Callable[[], None]) -> None:
    """绑定 config.HOTKEY -> callback（非阻塞注册）。"""
    keyboard.add_hotkey(config.HOTKEY, callback, suppress=False)


def wait() -> None:
    """阻塞主线程，直到 KeyboardInterrupt。"""
    try:
        keyboard.wait()
    except KeyboardInterrupt:
        pass
    finally:
        keyboard.unhook_all()
