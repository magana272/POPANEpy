from __future__ import annotations

from typing import Generator

import pandas as pd

from emotion.studies.metadata import POPANEMetadata

STUDY1_MEASUREMENT = {"TIMESTAMP", "AFFECT", "ECG",
                      "EDA", "TEMP", "RESPIRATION", "SBP", "DBP", "MARKER"}
STUDY2_MEASUREMENT = {"TIMESTAMP", "AFFECT", "ECG",
                      "EDA", "SBP", "DBP", "CO", "TPR", "MARKER"}
STUDY3_MEASUREMENT = {"TIMESTAMP", "AFFECT", "ECG",
                      "EDA", "SBP", "DBP", "CO", "TPR", "MARKER"}
STUDY4_MEASUREMENT = {"TIMESTAMP", "ECG",
                      "EDA", "SBP", "DBP", "CO", "TPR", "MARKER"}
STUDY5_MEASUREMENT = {"TIMESTAMP", "AFFECT", "ECG",
                      "EDA", "SBP", "DBP", "CO", "TPR", "MARKER"}
STUDY6_MEASUREMENT = {"TIMESTAMP", "AFFECT", "ECG", "DZDT",
                      "DZ", "Z0", "EDA", "SBP", "DBP", "CO", "TPR", "MARKER"}
STUDY7_MEASUREMENT = {"TIMESTAMP", "AFFECT",
                      "ECG", "DZDT", "DZ", "Z0", "MARKER"}


class Subject:
    """Represents a single subject in a study"""
    STUDY_MEASUEMENTS = {"STUDY1": STUDY1_MEASUREMENT,
                         "STUDY2": STUDY2_MEASUREMENT,
                         "STUDY3": STUDY3_MEASUREMENT,
                         "STUDY4": STUDY4_MEASUREMENT,
                         "STUDY5": STUDY5_MEASUREMENT,
                         "STUDY6": STUDY6_MEASUREMENT,
                         "STUDY7": STUDY7_MEASUREMENT}
    metadata: 'POPANEMetadata'
    data: 'pd.DataFrame | None'  # Generator[pd.DataFrame, Any, None] |

    def __init__(self, metadata: 'POPANEMetadata', data: pd.DataFrame | None = None) -> None:
        self.metadata = metadata
        self.data = None
        self.measurements = self.STUDY_MEASUEMENTS[list(
            set(metadata.STUDY_NAME))[0]]

    def __repr__(self):
        return f"Subject(study={self.metadata.STUDY_NAME}, id={self.metadata.SUBJECT_ID})"

    @property
    def emotions(self) -> pd.Series:
        """Get unique emotions for this subject"""
        if isinstance(self.data, Generator):
            return pd.Series(dtype=str)
        if self.data is None or 'EMOTION' not in self.data:
            return pd.Series(dtype=str)
        return pd.Series(self.data['EMOTION'].unique())

    def get_emotion_data(self, emotion: str) -> pd.DataFrame | None:
        """Filter data by emotion"""
        if self.data is None:
            return None
        if 'EMOTION' not in self.data:
            return None
        self.data = self.data[self.data['EMOTION'] == emotion]
        return self.data

    def get_measurements(self) -> set[str] | None:
        """Get specific measurements"""
        return self.measurements

    def to_dataframe(self) -> pd.DataFrame:
        """Convert subject data to a DataFrame"""
        if self.data is None:
            dfs = []
            if self.metadata.FILE_PATH is None:
                return pd.DataFrame()
            for file_name in self.metadata.FILE_PATH:
                df = pd.read_csv(file_name, skiprows=9)
                dfs.append(df)
            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            df.columns = df.columns.str.upper()
            self.data = df
        return self.data
