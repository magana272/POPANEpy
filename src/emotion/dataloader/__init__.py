from emotion.dataloader.popaneloader import POPANEDataLoader
from emotion.dataloader.downloader import POPANEDownloader

__version__ = "0.1.0"


def __getattr__(name):
    if name == "PopaneDataLoader":
        return POPANEDataLoader
    if name == "POPANEDownloader":
        return POPANEDownloader
    if name == "__version__":
        return __version__
    return None


__all__ = [
    "POPANEDataLoader",
    "POPANEDownloader"]
