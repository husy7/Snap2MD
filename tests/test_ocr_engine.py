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
    """_trim_tail_repetition 测试。"""

    def test_trims_repeating_digits(self):
        """末尾重复数字模式被截断。"""
        from ocr_engine import _trim_tail_repetition
        text = "栈内容\n1\n2\n1\n2\n1\n2\n1\n2\n1\n2\n"
        assert _trim_tail_repetition(text) == "栈内容"

    def test_trims_repeating_backticks(self):
        """末尾重复反引号被截断。"""
        from ocr_engine import _trim_tail_repetition
        bt = chr(96) * 3
        text = "内容" + bt * 50
        assert _trim_tail_repetition(text) == "内容"

    def test_normal_text_unchanged(self):
        """正常文本不被修改。"""
        from ocr_engine import _trim_tail_repetition
        text = "这是一段正常文本\n没有重复填充\n"
        assert _trim_tail_repetition(text) == text

    def test_empty_string(self):
        """空字符串原样返回。"""
        from ocr_engine import _trim_tail_repetition
        assert _trim_tail_repetition("") == ""

    def test_short_repetition_kept(self):
        """少量重复（<4次）不截断。"""
        from ocr_engine import _trim_tail_repetition
        text = "ababab"  # ab 重复 3 次，未达 4 次阈值
        assert _trim_tail_repetition(text) == text

    def test_trims_long_block_repetition(self):
        """末尾长段紧邻重复（>20字符）被截断。"""
        from ocr_engine import _trim_tail_repetition
        block = "Line1 with content\nLine2 with content\nLine3 with content"
        text = "Header\n" + block + "\n" + block
        assert _trim_tail_repetition(text) == "Header\n" + block

    def test_trims_reasoning_leakage_loop(self):
        """reasoning 泄露长段循环被截断（实测 glm-ocr 退化场景）。

        模拟用户实测：模型把英文 reasoning 泄露到输出，Wait 块循环多次。
        截断后保留 code + Actually（开场白）+ Wait（首次出现）。
        """
        from ocr_engine import _trim_tail_repetition
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
        assert _trim_tail_repetition(text) == code + "\n" + actually_block + "\n" + wait_block

    def test_trims_multiple_long_repetitions(self):
        """多次长段重复被递归全部截断。"""
        from ocr_engine import _trim_tail_repetition
        block = "First line of block\nSecond line of block\nThird line of block"
        text = "Header\n" + block + "\n" + block + "\n" + block + "\n" + block
        # 4 次重复，递归截断到只剩 1 次
        assert _trim_tail_repetition(text) == "Header\n" + block

    def test_normal_long_text_unchanged(self):
        """正常长文本（无末尾紧邻重复）不被修改。"""
        from ocr_engine import _trim_tail_repetition
        text = "\n".join([
            "# Title",
            "",
            "Paragraph one with unique content.",
            "",
            "Paragraph two with different content.",
            "",
            "Paragraph three concludes the text.",
        ])
        assert _trim_tail_repetition(text) == text


class TestStripReasoningLeakage:
    """_strip_reasoning_leakage 测试。"""

    def test_strips_actually_block(self):
        """单个 'Actually,' reasoning 块被整体移除，正文保留。"""
        from ocr_engine import _strip_reasoning_leakage
        code = "```python\ndef f():\n    return 1\n```"
        actually_block = (
            'Actually, the prompt says "recognize and output text content".\n'
            "The image shows a code snippet with math symbols. I'll just provide it as is.\n"
            "One detail: `matches = False` in line 5.\n"
            "I'll use LaTeX for that to match up exactly what I see on paper or screen."
        )
        text = code + "\n" + actually_block
        assert _strip_reasoning_leakage(text) == code

    def test_strips_wait_block(self):
        """单个 'Wait,' reasoning 块被移除。"""
        from ocr_engine import _strip_reasoning_leakage
        wait_block = (
            'Wait, the prompt says "recognize and output text content".\n'
            "The image shows a code snippet with math symbols."
        )
        text = "正文内容\n" + wait_block
        assert _strip_reasoning_leakage(text) == "正文内容"

    def test_strips_multiple_reasoning_blocks(self):
        """多个 reasoning 块（Actually + Wait）都被移除，code 保留。"""
        from ocr_engine import _strip_reasoning_leakage
        code = "```python\ndef f():\n    return 1\n```"
        actually_block = (
            'Actually, the prompt says "recognize and output text content".\n'
            "The image shows a code snippet with math symbols."
        )
        wait_block = (
            'Wait, the prompt says "recognize and output text content".\n'
            "The image shows a code snippet with math symbols."
        )
        text = code + "\n" + actually_block + "\n" + wait_block
        assert _strip_reasoning_leakage(text) == code

    def test_strips_reasoning_between_content(self):
        """reasoning 出现在两段正文之间：两段正文都保留，reasoning 移除。"""
        from ocr_engine import _strip_reasoning_leakage
        before = "# 标题\n这是第一段正文。"
        after = "这是第二段正文。"
        reasoning = (
            'Actually, I need to check the image again.\n'
            "Let me look at the code more carefully."
        )
        text = before + "\n" + reasoning + "\n" + after
        assert _strip_reasoning_leakage(text) == before + "\n" + after

    def test_keeps_reasoning_inside_code_fence(self):
        """代码栅栏内的 'Actually,' 不被剥离（保护真实代码）。"""
        from ocr_engine import _strip_reasoning_leakage
        text = "```python\nActually, this is real code\nprint('hello')\n```"
        assert _strip_reasoning_leakage(text) == text

    def test_keeps_chinese_content(self):
        """中文正文不被剥离（即使以 'Actually' 等词开头，含 CJK 即停）。"""
        from ocr_engine import _strip_reasoning_leakage
        text = "Actually 中文混合内容\n第二行"
        # 第一行含 CJK，不触发剥离
        assert _strip_reasoning_leakage(text) == text

    def test_keeps_markdown_table_after_reasoning(self):
        """reasoning 块遇到 Markdown 表格即停，表格保留。"""
        from ocr_engine import _strip_reasoning_leakage
        reasoning = (
            'Actually, let me think about this.\n'
            "I'll analyze the data first."
        )
        table = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        text = reasoning + "\n" + table
        assert _strip_reasoning_leakage(text) == table

    def test_stops_at_blank_line(self):
        """reasoning 块在空行处结束，空行后的正文保留。"""
        from ocr_engine import _strip_reasoning_leakage
        reasoning = (
            'Actually, the prompt says something.\n'
            "I'll think about it."
        )
        content = "正文内容"
        text = reasoning + "\n\n" + content
        assert _strip_reasoning_leakage(text) == content

    def test_keeps_normal_english_without_opener(self):
        """不以 reasoning opener 开头的英文正文保留。"""
        from ocr_engine import _strip_reasoning_leakage
        text = "The quick brown fox jumps over the lazy dog.\nSecond line of text."
        # "The quick" 不匹配 _REASONING_OPENER_RE（"The prompt"/"The image"/"The user" 才匹配）
        assert _strip_reasoning_leakage(text) == text

    def test_empty_string(self):
        """空字符串原样返回。"""
        from ocr_engine import _strip_reasoning_leakage
        assert _strip_reasoning_leakage("") == ""

    def test_strips_leading_reasoning_then_chinese(self):
        """reasoning 在开头、中文在后面：reasoning 移除，中文保留。"""
        from ocr_engine import _strip_reasoning_leakage
        reasoning = (
            'Hmm, the image shows a stack diagram.\n'
            "I'll output the numbers as they appear."
        )
        content = "栈内容\n1\n2\n3"
        text = reasoning + "\n" + content
        assert _strip_reasoning_leakage(text) == content


class TestStripIncompleteHtmlTables:
    """_strip_incomplete_html_tables 测试。"""

    def test_strips_trailing_incomplete_table(self):
        """末尾未闭合的 <table> 被移除，之前的正文保留。"""
        from ocr_engine import _strip_incomplete_html_tables
        content = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        incomplete = '<table border="1"><tr><td>truncated'
        text = content + "\n" + incomplete
        assert _strip_incomplete_html_tables(text) == content

    def test_keeps_complete_html_table(self):
        """完整的 <table>...</table> 不被移除（由 _convert_table 处理）。"""
        from ocr_engine import _strip_incomplete_html_tables
        text = '前文\n<table><tr><td>A</td></tr></table>\n后文'
        assert _strip_incomplete_html_tables(text) == text

    def test_no_html_unchanged(self):
        """无 HTML 表格的文本原样返回。"""
        from ocr_engine import _strip_incomplete_html_tables
        text = "纯文本内容\n第二行"
        assert _strip_incomplete_html_tables(text) == text

    def test_strips_multiple_incomplete_tables(self):
        """多个未闭合 <table> 迭代剥离。"""
        from ocr_engine import _strip_incomplete_html_tables
        content = "正文内容"
        text = content + "\n<table>incomplete1\n<table>incomplete2"
        assert _strip_incomplete_html_tables(text) == content

    def test_empty_string(self):
        """空字符串原样返回。"""
        from ocr_engine import _strip_incomplete_html_tables
        assert _strip_incomplete_html_tables("") == ""

    def test_real_world_sample_html_stripped(self):
        """实测样本：13 次重复 markdown 表格 + 末尾截断 HTML。

        _strip_incomplete_html_tables 仅负责移除 HTML 残片；
        13 次重复由后续 _trim_tail_repetition 处理。
        """
        from ocr_engine import _strip_incomplete_html_tables
        table = (
            "| Python中的递归深度限制 | Python中的递归深度限制 |\n"
            "| --- | --- |\n"
            "| ✓ | 在调试递归算法程序的时候经常会碰到这样的错误："
            "RecursionError递归的层数太多，系统调用栈容量有限 |"
        )
        html_fragment = (
            '<table border="1"><tr><td colspan="2">'
            'Python中的递归深度限制</td></ '
        )
        text = (table + "\n") * 13 + html_fragment
        result = _strip_incomplete_html_tables(text)
        # HTML 残片被移除
        assert '<table' not in result
        # 13 个表格完整保留（重复由后续 _trim_tail_repetition 处理）
        assert result.count("RecursionError") == 13

    def test_pipeline_strip_html_then_trim_repetition(self):
        """完整清洗管线：剥离 HTML 残片 -> 截末尾重复 -> 仅剩 1 个表格。"""
        from ocr_engine import _strip_incomplete_html_tables, _trim_tail_repetition
        table = (
            "| Python中的递归深度限制 | Python中的递归深度限制 |\n"
            "| --- | --- |\n"
            "| ✓ | 在调试递归算法程序的时候经常会碰到这样的错误："
            "RecursionError递归的层数太多，系统调用栈容量有限 |"
        )
        html_fragment = (
            '<table border="1"><tr><td colspan="2">'
            'Python中的递归深度限制</td></ '
        )
        text = (table + "\n") * 13 + html_fragment
        text = _strip_incomplete_html_tables(text)
        text = _trim_tail_repetition(text)
        assert text == table
