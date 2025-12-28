from emotion.utils.utils import get_size

__version__ = "0.1.0"


def __getattr__(name):
    if name == "ECG_SmoothTransformer":
        return get_size
    if name == "__version__":
        return __version__
    return None


__all__ = [
    "get_size", ]
