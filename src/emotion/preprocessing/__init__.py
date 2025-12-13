from emotion.preprocessing.dataprocess import ECG_SmoothTransformer


__version__ = "0.1.0"
def __getattr__(name):
    if name == "ECG_SmoothTransformer":
        return ECG_SmoothTransformer
    if name == "__version__":
        return __version__
    return None
__all__ = [
    "ECG_SmoothTransformer"]