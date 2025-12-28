from emotion.visualization.figure import POPANEFigureGenerator

__version__ = "0.1.0"


def __getattr__(name):
    if name == "POPANEFigureGenerator":
        return POPANEFigureGenerator
    if name == "__version__":
        return __version__
    return None


__all__ = [
    "POPANEFigureGenerator"]
