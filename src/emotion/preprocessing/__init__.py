from emotion.preprocessing.dataprocess import ECGSmoothTransformer

__version__ = "0.1.0"


def __getattr__(name):
    if name == "ECG_SmoothTransformer":
        return ECGSmoothTransformer
    if name == "__version__":
        return __version__
    return None


__all__ = [
    "ECGSmoothTransformer"]
