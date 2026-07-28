"""
反馈通知。

终端日志 + Windows toast 弹窗。
"""

from __future__ import annotations

import logging

import config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def success(char_count: int, elapsed: float) -> None:
    """完成通知。"""
    msg = f"\u2705 完成 | {char_count} 字 | {elapsed:.1f}s"
    logger.info(msg)
    _toast(f"\u2705 已复制 {char_count} 字")


def error(msg: str) -> None:
    """错误通知。"""
    logger.error(f"\u274c {msg}")
    _toast(f"\u274c {msg}")


def info(msg: str) -> None:
    """状态信息。"""
    logger.info(f"\u2139\ufe0f {msg}")


def _toast(msg: str) -> None:
    """Windows toast 弹窗（静默失败）。"""
    try:
        from plyer import notification

        notification.notify(
            title=config.TOAST_TITLE,
            message=msg,
            timeout=3,
        )
    except Exception:
        pass
