"""Ollama API 调用单元测试。"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import requests
from PIL import Image

import config
from ocr_engine import _encode, _call_api, recognize, warmup
from exceptions import OllamaConnectionError, OllamaTimeoutError, OllamaAPIError


class TestRecognize:
    """recognize 测试。"""

    @patch("ocr_engine._call_api")
    @patch("ocr_engine._encode")
    def test_normal(self, mock_encode, mock_call):
        """正常识别 -> 返回 content 字符串。"""
        mock_encode.return_value = "base64data"
        mock_call.return_value = {"message": {"content": "# Title\ncontent"}}
        img = Image.new("RGB", (100, 50))
        result = recognize(img)
        assert result == "# Title\ncontent"

    @patch("ocr_engine.requests.post")
    def test_connection_error(self, mock_post):
        """连接失败 -> 抛 OllamaConnectionError。"""
        mock_post.side_effect = requests.exceptions.ConnectionError()
        with pytest.raises(OllamaConnectionError):
            recognize(Image.new("RGB", (10, 10)))

    @patch("ocr_engine.requests.post")
    def test_timeout_error(self, mock_post):
        """超时 -> 抛 OllamaTimeoutError。"""
        mock_post.side_effect = requests.exceptions.Timeout()
        with pytest.raises(OllamaTimeoutError):
            recognize(Image.new("RGB", (10, 10)))

    @patch("ocr_engine.requests.post")
    def test_non_200(self, mock_post):
        """非 200 响应 -> 抛 OllamaAPIError。"""
        mock_response = type("Resp", (), {"status_code": 500})()
        mock_post.return_value = mock_response
        with pytest.raises(OllamaAPIError):
            recognize(Image.new("RGB", (10, 10)))

    @patch("ocr_engine.requests.post")
    def test_no_content_in_response(self, mock_post):
        """JSON 无 content -> 抛 OllamaAPIError。"""
        mock_response = type("Resp", (), {"status_code": 200})()
        mock_response.json = lambda: {"message": {}}
        mock_post.return_value = mock_response
        with pytest.raises(OllamaAPIError):
            recognize(Image.new("RGB", (10, 10)))


class TestEncode:
    """_encode 测试。"""

    def test_large_image_scaled(self):
        """大图压缩。"""
        img = Image.new("RGB", (4000, 3000))
        b64 = _encode(img)
        assert isinstance(b64, str) and len(b64) > 0

    def test_small_image_unchanged(self):
        """小图不压缩。"""
        img = Image.new("RGB", (800, 600))
        b64 = _encode(img)
        assert isinstance(b64, str) and len(b64) > 0

    def test_rgba_to_rgb(self):
        """RGBA 转 RGB -> JPEG 编码。"""
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        b64 = _encode(img)
        assert isinstance(b64, str) and len(b64) > 0


class TestCallApiPayload:
    """_call_api payload 字段测试。"""

    @patch("ocr_engine.requests.post")
    def test_payload_includes_num_ctx_and_keep_alive(self, mock_post):
        """payload 包含 options.num_ctx 和 keep_alive。"""
        mock_response = type("Resp", (), {"status_code": 200})()
        mock_response.json = lambda: {"message": {"content": "ok"}}
        mock_post.return_value = mock_response

        _call_api("base64data")

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["options"]["num_ctx"] == config.NUM_CTX
        assert payload["options"]["repeat_penalty"] == config.REPEAT_PENALTY
        assert payload["options"]["num_predict"] == config.NUM_PREDICT
        assert payload["keep_alive"] == config.KEEP_ALIVE
        assert payload["model"] == config.MODEL
        assert payload["messages"][0]["content"] == config.PROMPT
        assert payload["stream"] is False

    @patch("ocr_engine.requests.post")
    def test_payload_hits_chat_endpoint(self, mock_post):
        """_call_api 命中 /api/chat 端点。"""
        mock_response = type("Resp", (), {"status_code": 200})()
        mock_response.json = lambda: {"message": {"content": "ok"}}
        mock_post.return_value = mock_response

        _call_api("base64data")

        args, _ = mock_post.call_args
        assert args[0].endswith("/api/chat")


class TestWarmup:
    """warmup 测试。"""

    @patch("ocr_engine.requests.post")
    def test_warmup_hits_generate_endpoint(self, mock_post):
        """warmup 命中 /api/generate 端点。"""
        mock_post.return_value = type("Resp", (), {"status_code": 200})()
        warmup()
        args, _ = mock_post.call_args
        assert args[0].endswith("/api/generate")

    @patch("ocr_engine.requests.post")
    def test_warmup_payload_keep_alive_and_num_ctx(self, mock_post):
        """warmup payload 包含 keep_alive 和 options.num_ctx。"""
        mock_post.return_value = type("Resp", (), {"status_code": 200})()
        warmup()
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["keep_alive"] == config.KEEP_ALIVE
        assert payload["options"]["num_ctx"] == config.NUM_CTX
        assert payload["options"]["repeat_penalty"] == config.REPEAT_PENALTY
        assert payload["options"]["num_predict"] == config.NUM_PREDICT
        assert payload["model"] == config.MODEL

    @patch("ocr_engine.requests.post")
    def test_warmup_silent_on_connection_error(self, mock_post):
        """Ollama 未启动时 warmup 静默不抛异常。"""
        mock_post.side_effect = requests.exceptions.ConnectionError()
        warmup()  # 不应抛异常

    @patch("ocr_engine.requests.post")
    def test_warmup_silent_on_timeout(self, mock_post):
        """warmup 超时时静默不抛异常。"""
        mock_post.side_effect = requests.exceptions.Timeout()
        warmup()  # 不应抛异常


class TestTrimRepetition:
    """_trim_repetition 测试。"""

    def test_trims_repeating_digits(self):
        """末尾重复数字模式被截断。"""
        from ocr_engine import _trim_repetition
        text = "栈内容\n1\n2\n1\n2\n1\n2\n1\n2\n1\n2\n"
        assert _trim_repetition(text) == "栈内容"

    def test_trims_repeating_backticks(self):
        """末尾重复反引号被截断。"""
        from ocr_engine import _trim_repetition
        bt = chr(96) * 3
        text = "内容" + bt * 50
        assert _trim_repetition(text) == "内容"

    def test_normal_text_unchanged(self):
        """正常文本不被修改。"""
        from ocr_engine import _trim_repetition
        text = "这是一段正常文本\n没有重复填充\n"
        assert _trim_repetition(text) == text

    def test_empty_string(self):
        """空字符串原样返回。"""
        from ocr_engine import _trim_repetition
        assert _trim_repetition("") == ""

    def test_short_repetition_kept(self):
        """少量重复（<4次）不截断。"""
        from ocr_engine import _trim_repetition
        text = "ababab"  # ab 重复 3 次，未达 4 次阈值
        assert _trim_repetition(text) == text

    def test_trims_long_block_repetition(self):
        """末尾长段紧邻重复（>20字符）被截断。"""
        from ocr_engine import _trim_repetition
        block = "Line1 with content\nLine2 with content\nLine3 with content"
        text = "Header\n" + block + "\n" + block
        assert _trim_repetition(text) == "Header\n" + block

    def test_trims_reasoning_leakage_loop(self):
        """reasoning 泄露长段循环被截断（实测 glm-ocr 退化场景）。

        模拟用户实测：模型把英文 reasoning 泄露到输出，Wait 块循环多次。
        截断后保留 code + Actually（开场白）+ Wait（首次出现）。
        """
        from ocr_engine import _trim_repetition
        code = "```python\ndef f():\n    return 1\n```"
        actually_block = (
            'Actually, the prompt says "recognize and output text content".\n'
            "The image shows a code snippet with math symbols. I'll just provide it as is.\n"
            "One detail: `matches = False` in line 5.\n"
            "I'll use LaTeX for that to match up exactly what I see on paper or screen (which might be different)."
        )
        wait_block = (
            'Wait, the prompt says "recognize and output text content".\n'
            "The image shows a code snippet with math symbols. I'll just provide it as is.\n"
            "One detail: `matches = False` in line 5.\n"
            "I'll use LaTeX for that to match up exactly what I see on paper or screen (which might be different)."
        )
        text = code + "\n" + actually_block + "\n" + wait_block + "\n" + wait_block + "\n" + wait_block
        assert _trim_repetition(text) == code + "\n" + actually_block + "\n" + wait_block

    def test_trims_multiple_long_repetitions(self):
        """多次长段重复被递归全部截断。"""
        from ocr_engine import _trim_repetition
        block = "First line of block\nSecond line of block\nThird line of block"
        text = "Header\n" + block + "\n" + block + "\n" + block + "\n" + block
        # 4 次重复，递归截断到只剩 1 次
        assert _trim_repetition(text) == "Header\n" + block

    def test_normal_long_text_unchanged(self):
        """正常长文本（无末尾紧邻重复）不被修改。"""
        from ocr_engine import _trim_repetition
        text = "\n".join([
            "# Title",
            "",
            "Paragraph one with unique content.",
            "",
            "Paragraph two with different content.",
            "",
            "Paragraph three concludes the text.",
        ])
        assert _trim_repetition(text) == text
