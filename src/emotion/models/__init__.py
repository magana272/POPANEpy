from emotion.models.random_forest import EmotionRandomForest

__version__ = "0.1.0"


def __getattr__(name):
    if name == "EmotionRandomForest":
        return EmotionRandomForest
    if name == "__version__":
        return __version__
    return None


__all__ = [
    "EmotionRandomForest"]
