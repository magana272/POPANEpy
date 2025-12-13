"""
POPANE emotion study implementations
"""
from dataclasses import dataclass
import re
from typing import Any, Generator, Optional, ClassVar, TYPE_CHECKING
from numpy.f2py.symbolic import Op
import pandas as pd
from polars import config
from emotion.dataloader.popaneloader import POPANEDataLoader


@dataclass
class Subject:
    """Represents a single subject in a study"""
    STUDY_NAME: 'Study'
    SUBJECT_ID: int
    FILE_NAME: str
    FILE_PATH: str
    AGE: int
    SEX: int
    EMOTION: str

    data: pd.DataFrame | Generator[pd.DataFrame, Any, None]

    def __repr__(self):
        return f"Subject(study={self.STUDY_NAME.config.number}, id={self.SUBJECT_ID})"

    @property
    def emotions(self) -> pd.Series:
        """Get unique emotions for this subject"""
        if isinstance(self.data, Generator):
            return pd.Series(dtype=str)
        if 'EMOTION' not in self.data:
            return pd.Series(dtype=str)
        return pd.Series(self.data['EMOTION'].unique())

    def get_emotion_data(self, emotion: str) -> pd.DataFrame:
        """Filter data by emotion"""
        if isinstance(self.data, Generator):
            return pd.DataFrame()
        return self.data[self.data['EMOTION'] == emotion]

    def get_measurements(self, measurements: list[str]) -> pd.DataFrame:
        """Get specific measurements"""
        if isinstance(self.data, Generator):
            return pd.DataFrame()
        return self.data[measurements]


@dataclass
class StudyConfig:
    """Configuration for a POPANE study"""
    number: int
    name: str
    measurements: tuple[str, ...]
    dtypes: dict[str, str]

    @property
    def columns(self) -> set[str]:
        return set(self.measurements)


class Study:
    """Represents a POPANE emotion study with multiple subjects"""

    config: ClassVar[StudyConfig]

    def __init__(self, data_loader: 'POPANEDataLoader'):
        self.loader = data_loader
        self._subjects: dict[int, 'Subject'] = {}

    def get_subject(self, subject_id: int, emotion: str) -> 'Subject':
        if subject_id not in self._subjects:
            df = self.loader.get_data_for_subject_from_study(
                self.config.number, subject_id
            )
            study_meta = self.loader.get_study_metadata(self.config.number)

            if study_meta is None:
                raise ValueError(
                    f"Study metadata for Study {self.config.number} not found.")
            subject_meta = study_meta[(study_meta['SUBJECT_ID'] == subject_id) & (
                study_meta['EMOTION'] == emotion)]
            print(f"Subject Meta:\n{subject_meta}")
            if df is None:
                raise ValueError(
                    f"Subject {subject_id} not found in Study {self.config.number}")
            self._subjects[subject_id] = Subject(
                STUDY_NAME=self,
                SUBJECT_ID=subject_id,
                FILE_NAME=subject_meta.FILE_NAME.values[0] if not subject_meta.empty else '',
                FILE_PATH=subject_meta.FILE_PATH.values[0] if not subject_meta.empty else '',
                AGE=subject_meta.AGE.values[0] if not subject_meta.empty else 0,
                SEX=subject_meta.SEX.values[0] if not subject_meta.empty else 0,
                EMOTION=emotion,
                data=df
            )
        return self._subjects[subject_id]

    def get_subject_lazy(self, subject_id: int, measurements: tuple[str, ...],
                         dtypes: dict[str, str], emotion: str) -> 'Subject':
        df = self.loader.get_data_for_subject_from_study_lazy(
            self.config.number, str(subject_id), measurements, dtypes
        )
        study_meta = self.loader.get_study_metadata(self.config.number)

        if study_meta is None:
            raise ValueError(
                f"Study metadata for Study {self.config.number} not found.")
        subject_meta = study_meta[study_meta['SUBJECT_ID']
                                  == subject_id].iloc[0]
        if df is None:
            raise ValueError(
                f"Subject {subject_id} not found in Study {self.config.number}")
        self._subjects[subject_id] = Subject(
            STUDY_NAME=self,
            SUBJECT_ID=subject_id,
            FILE_NAME=subject_meta.get('FILE_NAME', ''),
            FILE_PATH=subject_meta.get('FILE_PATH', ''),
            AGE=subject_meta.get('AGE', 0),
            SEX=subject_meta.get('SEX', 0),
            EMOTION=emotion,
            data=df
        )
        return self._subjects[subject_id]

    def get_measurements(self) -> dict[str, str]:
        return self.config.dtypes

    def get_all_subjects(self) -> list[Optional['Subject']]:
        subject_ids = self.loader.get_subject_ids(self.config.number)
        metadata = self.loader.get_study_metadata(self.config.number)
        res = []
        for sid in subject_ids:
            if metadata is not None:
                subject_meta = metadata[metadata['SUBJECT_ID'] == sid]
                if subject_meta.empty:
                    continue
                emotions = subject_meta['EMOTION'].unique()
                for emotion in emotions:
                    res.append(self.get_subject_lazy(sid, self.config.measurements,
                                                     self.config.dtypes, emotion=emotion))
        return res

    @property
    def available_measurements(self) -> tuple[str, ...]:
        return self.config.measurements


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


# Registry for easy access
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
