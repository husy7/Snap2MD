"""剪贴板读写单元测试。"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from PIL import Image

from clipboard_io import read_image, write_text
from exceptions import ClipboardError


class TestReadImage:
    """read_image 测试。"""

    @patch("clipboard_io.ImageGrab.grabclipboard")
    def test_clipboard_has_image(self, mock_grab):
        """剪贴板有图片 -> 返回 Image 对象。"""
        img = Image.new("RGB", (100, 50))
        mock_grab.return_value = img
        result = read_image()
        assert result is img

    @patch("clipboard_io.ImageGrab.grabclipboard")
    def test_clipboard_file_paths(self, mock_grab, tmp_path):
        """剪贴板为文件路径 -> 读取文件返回 Image。"""
        file_path = tmp_path / "shot.png"
        img = Image.new("RGB", (50, 50))
        img.save(file_path)
        mock_grab.return_value = [str(file_path)]
        result = read_image()
        assert result is not None
        assert result.size == (50, 50)

    @patch("clipboard_io.ImageGrab.grabclipboard")
    def test_clipboard_empty(self, mock_grab):
        """剪贴板为空 -> 返回 None。"""
        mock_grab.return_value = None
        assert read_image() is None

    @patch("clipboard_io.ImageGrab.grabclipboard")
    def test_clipboard_text(self, mock_grab):
        """剪贴板为纯文本 -> 返回 None。"""
        mock_grab.return_value = "hello"
        assert read_image() is None


class TestWriteText:
    """write_text 测试。"""

    @patch("clipboard_io.pyperclip.copy")
    def test_write_normal(self, mock_copy):
        """正常写入 -> pyperclip.copy 被调用 1 次。"""
        write_text("hello")
        mock_copy.assert_called_once_with("hello")

    @patch("clipboard_io.pyperclip.copy")
    def test_write_retry_success(self, mock_copy):
        """首次失败第 2 次成功 -> 2 次调用最终成功。"""
        mock_copy.side_effect = [RuntimeError("locked"), None]
        write_text("hello")
        assert mock_copy.call_count == 2

    @patch("clipboard_io.pyperclip.copy")
    def test_write_all_fail(self, mock_copy):
        """连续 3 次失败 -> 抛 ClipboardError。"""
        mock_copy.side_effect = RuntimeError("locked")
        with pytest.raises(ClipboardError):
            write_text("hello")
        assert mock_copy.call_count == 3
