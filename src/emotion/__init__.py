"""
POPANE Emotion - Package for emotion study data analysis
"""
from emotion.studies.study import Study, get_study, Subject
from emotion.POPANE import POPANE
from emotion.db.db import POPANEDB

from emotion.utils.utils import get_size
from emotion.visualization.figure import POPANEFigureGenerator

__version__ = "0.1.0"


def __getattr__(name):
    if name == "POPANEFigureGenerator":
        return POPANEFigureGenerator
    if name == "get_size":
        return get_size
    if name == "Study":
        return Study
    if name == "Subject":
        return Subject
    if name == "get_study":
        return get_study
    if name == "POPANE":
        return POPANE
    if name == "PROPANEDB":
        return POPANEDB
    if name == "__version__":
        return __version__
    return None


__all__ = [
    "POPANEFigureGenerator",
    "get_size",
    "Study",
    "Subject",
    "get_study",
    "POPANE",
]
