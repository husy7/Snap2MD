"""
自定义异常层次。
"""


class AppError(Exception):
    """应用基类异常。"""


class ClipboardError(AppError):
    """剪贴板读写异常。"""


class OllamaConnectionError(AppError):
    """Ollama 连接失败。"""


class OllamaTimeoutError(AppError):
    """Ollama 请求超时。"""


class OllamaAPIError(AppError):
    """Ollama API 返回异常。"""
