
from emotion.dataloader.popaneloader import POPANEDataLoader
__version__ = "0.1.0"


def __getattr__(name):
    if name == "PopaneDataLoader":
        return POPANEDataLoader
    if name == "__version__":
        return __version__
    return None
__all__ = [
    "POPANEDataLoader"]
