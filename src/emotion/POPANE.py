import numpy as np
import numpy.typing as npt
import pandas as pd
from polars import DataFrame

from emotion.dataloader.popaneloader import POPANEDataLoader
from emotion.studies import Study1, Study2, Study3, Study4, Study5, Study6, Study7, Study
from emotion.studies import Subject
from emotion.studies.subject import POPANEMetadata


class POPANE:
    studies: dict[int, Study | None] = {1: None,
                                        2: None,
                                        3: None,
                                        4: None,
                                        5: None,
                                        6: None,
                                        7: None}

    def __init__(self):
        self.loader = POPANEDataLoader(data_dir="data/raw/")
        self.studies[1] = Study1(data_loader=self.loader)
        self.studies[2] = Study2(data_loader=self.loader)
        self.studies[3] = Study3(data_loader=self.loader)
        self.studies[4] = Study4(data_loader=self.loader)
        self.studies[5] = Study5(data_loader=self.loader)
        self.studies[6] = Study6(data_loader=self.loader)
        self.studies[7] = Study7(data_loader=self.loader)

    def get_unique_emotions(self, study_number) -> list[str]:
        """Get a list of unique emotions present in the specified study."""
        study = self.studies.get(study_number)
        if study is None:
            return []
        return study.get_unique_emotions()

    def get_subject_ids(self, study_number) -> list[int]:
        """Get a list of subject IDs for a specific study."""
        study = self.studies.get(study_number)
        if study is None:
            return []
        return study.get_subject_ids()

    def get_study_metadata(self, study_number) -> POPANEMetadata | None:
        """Retrieve metadata for a specific study based on the study number."""
        study = self.studies.get(study_number)
        if study is None:
            return None
        return study.get_study_metadata()

    def get_data_for_subject_from_study(self, study_number, subject_id) -> pd.DataFrame | None:
        """Get data for a specific subject from a specified study."""
        study = self.studies.get(study_number)
        if study is None:
            return None
        subject = study.get_subject(subject_id, emotion=None).to_dataframe()
        return subject

    def get_unique_emotions_for_all_studies(self):
        """Get a list of all unique 
        emotions across all studies.
        """
        all_unique_emotions = set()
        for study_number in range(1, 8):
            unique_emotions = self.get_unique_emotions(study_number)
            all_unique_emotions |= set(unique_emotions)
        return list(set(all_unique_emotions))

    def get_data_from_study_by_emotions(self, study_number, emotions) -> DataFrame | None:
        """Get data from a specified study for given emotions."""
        study = self.studies.get(study_number)
        if study is None:
            return None
        return study.get_from_study_by_emotion(emotions)

    def get_overlap_features_across_studies(self, study_numbers: npt.NDArray[int] | list[int]) -> npt.NDArray[str]:
        """Get overlapping features across specified studies."""
        set_list: list[set[str] | None] = [
            self.get_features_for_study(num) for num in study_numbers]
        valid_sets = [s for s in set_list if s is not None]
        if not valid_sets:
            return set()
        overlap_features = set.intersection(*valid_sets)
        return np.array(overlap_features)

    def get_features_for_study(self, study_number: int) -> set[str] | None:
        """Get features for a specific study."""
        study = self.studies.get(study_number)
        if study is None:
            return None
        measurements = study.get_measurements()
        if measurements is None:
            return None
        return set(list(measurements.keys()))

    def get_all_features(self) -> npt.NDArray:
        """Get all unique features across all studies."""
        all_features = set()
        for study_number in range(1, 8):
            features = self.get_features_for_study(study_number)
            if features is not None:
                all_features.update(features)
        return np.array(list(all_features))

    def get_subject_metadata_by_id(self, study_number: int, subject_id: int) -> Subject | None:
        """Retrieve subject metadata for a specific study based on the study number."""
        study = self.studies.get(study_number)
        if study is None:
            return None
        return study.get_subject(subject_id, emotion=None)
