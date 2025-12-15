"""
POPANE emotion studies - Study and Subject classes
"""
__version__ = "0.1.0"

from emotion.studies.study import (
    Study, StudyConfig,
    Study1, Study2, Study3, Study4, Study5, Study6, Study7,
    STUDY_REGISTRY, get_study
)
from emotion.studies.study_config import (StudyConfig)
from emotion.studies.subject import (
    Subject
)


def __getattr__(name):
    if name == "Study":
        return Study
    elif name == "Study1":
        return Study1
    elif name == "Study2":
        return Study2
    elif name == "Study3":
        return Study3
    elif name == "Study4":
        return Study4
    elif name == "Study5":
        return Study5
    elif name == "Study6":
        return Study6
    elif name == "Study7":
        return Study7
    elif name == "Subject":
        return Subject
    elif name == "STUDY_REGISTRY":
        return STUDY_REGISTRY
    if name == "__version__":
        return __version__
    return None


__all__ = [
    "Study",
    "StudyConfig",
    "Subject",
    "Study1",
    "Study2",
    "Study3",
    "Study4",
    "Study5",
    "Study6",
    "Study7",
    "STUDY_REGISTRY",
    "get_study",
]
