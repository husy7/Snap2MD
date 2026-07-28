"""集成测试。

边界场景测试，验证各模块协同工作的健壮性。
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from PIL import Image

import main
from clipboard_io import read_image


class TestIntegration:
    """边界场景集成测试。"""

    def test_white_image(self):
        """纯白图片 -> 不崩溃。"""
        img = Image.new("RGB", (100, 100), (255, 255, 255))
        with patch("clipboard_io.ImageGrab.grabclipboard", return_value=img):
            result = read_image()
            assert result is not None

    def test_large_image(self):
        """超大图片 -> 自动压缩后正常处理。"""
        img = Image.new("RGB", (8000, 6000))
        with patch("clipboard_io.ImageGrab.grabclipboard", return_value=img):
            result = read_image()
            assert result is not None

    def test_tiny_image(self):
        """极小图片 -> 不崩溃。"""
        img = Image.new("RGB", (10, 10))
        with patch("clipboard_io.ImageGrab.grabclipboard", return_value=img):
            result = read_image()
            assert result is not None

    def test_no_image_in_clipboard(self):
        """非图片剪贴板 -> read_image 返回 None。"""
        with patch("clipboard_io.ImageGrab.grabclipboard", return_value=None):
            assert read_image() is None

    def test_concurrent_trigger_ignored(self):
        """防抖测试 -> 锁被持有时忽略触发。"""
        main._lock.acquire()
        try:
            main.pipeline()
        finally:
            main._lock.release()

    @pytest.fixture(autouse=True)
    def _reset_lock(self):
        """每个测试后重置防抖锁。"""
        yield
        try:
            main._lock.release()
        except RuntimeError:
            pass
