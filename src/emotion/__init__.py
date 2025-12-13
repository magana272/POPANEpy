"""
POPANE Emotion - Package for emotion study data analysis
"""
from math import e
import re
from emotion.visualization.figure import POPANEFigureGenerator
from emotion.dataloader import PopaneDataLoader
from emotion.utils.utils import get_size
from emotion.studies.study import Study, get_study, Subject
__version__ = "0.1.0"

def __getattr__(name):
    if name == "PopaneDataLoader":
        return PopaneDataLoader
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
    if name == "__version__":
        return __version__
    return None
__all__ = [
    "PopaneDataLoader",
    "POPANEFigureGenerator",
    "get_size",
    "Study",
    "Subject",
    "get_study",
]
