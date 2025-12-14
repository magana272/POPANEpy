"""
POPANE emotion study implementations
"""
from dataclasses import dataclass
from typing import ClassVar
import pandas as pd
from emotion.studies.dataloader.popaneloader import POPANEDataLoader
from emotion.studies.subject import Subject, POPANEMetadata


@dataclass
class StudyConfig:
    """Configuration for a POPANE study"""
    number: int
    name: str
    measurements: tuple[str, ...]
    dtypes: dict[str, str] | None

    @property
    def columns(self) -> set[str]:
        return set(self.measurements)


class Study:
    """Represents a POPANE emotion study with multiple subjects"""

    config: ClassVar[StudyConfig]

    def __init__(self, data_loader: 'POPANEDataLoader'):
        self.loader = data_loader
        self._subjects: dict[int, 'Subject'] = {}

    def get_subject(self, subject_id: int, emotion: str | None = None) -> Subject | None:
        if subject_id not in self._subjects:
            df = self.loader.get_subject_df(self.config.number, subject_id)
            subject_meta = self.loader.get_subject_metadata(
                self.config.number, subject_id)
            if subject_meta is None:
                raise ValueError(
                    f"Study metadata for Study {self.config.number} not found.")
            return Subject(subject_meta)

    def get_subject_ids(self) -> list[int]:
        return self.loader.get_subject_ids(self.config.number)

    def get_study_metadata(self) -> POPANEMetadata | None:
        return self.loader.get_study_metadata(self.config.number)

    def get_unique_emotions(self) -> list[str]:
        return self.loader.get_unique_emotions(self.config.number)

    def get_measurements(self) -> dict[str, str] | None:
        return self.config.dtypes

    def get_all_subjects(self, study_numbers: list[int]) -> dict[int, Subject | None]:
        return self.loader.get_all_subjects_from_study(self.config.number, self.get_unique_emotions())
    def get_all_subjects_from_study(self, study_number: int, emotions: list[str]) -> dict[int, Subject | None]:
        return self.loader.get_all_subjects_from_study(study_number, emotions)

    @property
    def available_measurements(self) -> tuple[str, ...]:
        return self.config.measurements

    def __repr__(self):
        return f"Study(number={self.config.number}, name={self.config.name})"

    def __str__(self):
        return f"Study {self.config.number}: {self.config.name}"

    def __len__(self):
        return len(self.get_subject_ids())


class Study1(Study):
    config = StudyConfig(
        number=1,
        name="study1",
        measurements=("timestamp", "affect", "ECG", "EDA", "temp",
                      "respiration", "SBP", "DBP", "marker"),
        dtypes={
            "timestamp": "float", "affect": "float", "ECG": "float",
            "EDA": "float", "temp": "float", "respiration": "float",
            "SBP": "float", "DBP": "float", "marker": "int"
        }
    )


class Study2(Study):
    config = StudyConfig(
        number=2,
        name="study2",
        measurements=("timestamp", "affect", "ECG", "EDA", "SBP",
                      "DBP", "CO", "TPR", "marker"),
        dtypes={
            "timestamp": "float", "affect": "float", "ECG": "float",
            "EDA": "float", "SBP": "float", "DBP": "float",
            "CO": "float", "TPR": "float", "marker": "int"
        }
    )


class Study3(Study):
    config = StudyConfig(
        number=3,
        name="study3",
        measurements=("timestamp", "affect", "ECG", "EDA", "SBP",
                      "DBP", "CO", "TPR", "marker"),
        dtypes={
            "timestamp": "float", "affect": "float", "ECG": "float",
            "EDA": "float", "SBP": "float", "DBP": "float",
            "CO": "float", "TPR": "float", "marker": "int"
        }
    )


class Study4(Study):
    config = StudyConfig(
        number=4,
        name="study4",
        measurements=("timestamp", "ECG", "EDA", "SBP",
                      "DBP", "CO", "TPR", "marker"),
        dtypes={
            "timestamp": "float", "ECG": "float", "EDA": "float",
            "SBP": "float", "DBP": "float", "CO": "float",
            "TPR": "float", "marker": "int"
        }
    )


class Study5(Study):
    config = StudyConfig(
        number=5,
        name="study5",
        measurements=("timestamp", "affect", "ECG", "EDA", "SBP",
                      "DBP", "CO", "TPR", "marker"),
        dtypes={
            "timestamp": "float", "affect": "float", "ECG": "float",
            "EDA": "float", "SBP": "float", "DBP": "float",
            "CO": "float", "TPR": "float", "marker": "int"
        }
    )


class Study6(Study):
    config = StudyConfig(
        number=6,
        name="study6",
        measurements=("timestamp", "affect", "ECG", "dzdt", "dz", "z0",
                      "EDA", "SBP", "DBP", "CO", "TPR", "marker"),
        dtypes={
            "timestamp": "float", "affect": "float", "ECG": "float",
            "dzdt": "float", "dz": "float", "z0": "float", "EDA": "float",
            "SBP": "float", "DBP": "float", "CO": "float",
            "TPR": "float", "marker": "int"
        }
    )


class Study7(Study):
    config = StudyConfig(
        number=7,
        name="study7",
        measurements=("timestamp", "affect", "ECG",
                      "dzdt", "dz", "z0", "marker"),
        dtypes={
            "timestamp": "float", "affect": "float", "ECG": "float",
            "dzdt": "float", "dz": "float", "z0": "float", "marker": "int"
        }
    )


STUDY_REGISTRY = {
    1: Study1,
    2: Study2,
    3: Study3,
    4: Study4,
    5: Study5,
    6: Study6,
    7: Study7,
}


def get_study(study_number: int, data_loader) -> Study:
    if study_number not in STUDY_REGISTRY:
        raise ValueError(
            f"Study {study_number} not found. Available: {list(STUDY_REGISTRY.keys())}")
    return STUDY_REGISTRY[study_number](data_loader)
