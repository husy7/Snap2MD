"""
入口 + 主流程编排。

不含业务逻辑，仅编排各模块。
"""

from __future__ import annotations

import threading
import time

import clipboard_io
import config
import hotkey
import notifier
import ocr_engine
from exceptions import AppError

_lock = threading.Lock()
_last_result: str = ""     # 跨次去重：上次写入剪贴板的结果


def pipeline() -> None:
    """单次识别流程（由快捷键触发）。"""
    if not _lock.acquire(blocking=False):
        return

    try:
        start = time.time()

        img = clipboard_io.read_image()
        if img is None:
            notifier.error("剪贴板无图片")
            return

        raw = ocr_engine.recognize(img)

        if not raw.strip():
            notifier.error("未识别到内容")
            return

        global _last_result
        if raw == _last_result:
            notifier.info("结果与上次相同，跳过写入")
            return
        _last_result = raw

        clipboard_io.write_text(raw)
        elapsed = time.time() - start
        notifier.success(len(raw), elapsed)

    except AppError as e:
        notifier.error(str(e))
    finally:
        _lock.release()


def main() -> None:
    """主入口。"""
    notifier.info(f"已启动，快捷键 {config.HOTKEY}，等待截图...")
    threading.Thread(target=ocr_engine.warmup, daemon=True).start()
    hotkey.register(pipeline)
    hotkey.wait()


if __name__ == "__main__":
    main()
