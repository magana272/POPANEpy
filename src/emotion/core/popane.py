"""
POPANE - Main API Facade for POPANE Emotion Study Analysis

This module provides a unified interface for accessing emotion study data,
managing multiple studies, and performing cross-study analyses.
"""
import pandas as pd

from emotion.dataloader.popaneloader import POPANEDataLoader
from emotion.studies import Study1, Study2, Study3, Study4, Study5, Study6, Study7, Study
from emotion.studies.subject import POPANEMetadata, Subject


class POPANE:
    """
    Main facade class for POPANE emotion study data analysis.

    Provides high-level API for:

    - Loading and accessing study data
    - Managing multiple study instances
    - Querying subjects and metadata
    - Cross-study feature analysis
    Example:
        popane = POPANE()
        emotions = popane.get_unique_emotions(1)
        ubject = popane.get_subject(1, 123)
    """

    def __init__(self, data_dir: str = "data/raw/"):
        self.loader = POPANEDataLoader(data_dir=data_dir)
        self.studies: dict[int, Study] = {
            1: Study1(data_loader=self.loader),
            2: Study2(data_loader=self.loader),
            3: Study3(data_loader=self.loader),
            4: Study4(data_loader=self.loader),
            5: Study5(data_loader=self.loader),
            6: Study6(data_loader=self.loader),
            7: Study7(data_loader=self.loader)
        }


    def get_study(self, study_number: int) -> Study | None:
        """Get a study instance by number (1-7)."""
        return self.studies.get(study_number)

    def get_unique_emotions(self, study_number: int) -> list[str]:
        """Get a list of unique emotions present in the specified study."""
        study = self.studies.get(study_number)
        if study is None:
            return []
        return study.get_unique_emotions()

    def get_unique_emotions_for_all_studies(self) -> list[str]:
        """Get all unique emotions across all 7 studies."""
        all_unique_emotions = set()
        for study_number in range(1, 8):
            unique_emotions = self.get_unique_emotions(study_number)
            all_unique_emotions.update(unique_emotions)
        return sorted(all_unique_emotions)

    def get_subject_ids(self, study_number: int) -> list[int]:
        """Get a list of subject IDs for a specific study."""
        study = self.studies.get(study_number)
        if study is None:
            return []
        return study.get_subject_ids()

    def get_subject(self, study_number: int, subject_id: int) -> Subject | None:
        study = self.studies.get(study_number)
        if study is None:
            return None
        return study.get_subject(subject_id, emotion=None)

    def get_study_metadata(self, study_number: int) -> POPANEMetadata | None:
        """Retrieve metadata for a specific study."""
        study = self.studies.get(study_number)
        if study is None:
            return None
        return study.get_study_metadata()

    def get_subject_metadata(self, study_number: int, subject_id: int) -> Subject | None:
        """Retrieve subject metadata for a specific study and subject ID."""
        return self.get_subject(study_number, subject_id)

    def get_subjects_by_emotion(self, study_number: int, emotions: list[str]) -> dict[int, Subject | None]:
        """
        Get all subjects from a study filtered by emotion.
        
        Args:
            study_number: Study number (1-7)
            emotions: List of emotions to filter by
            
        Returns:
            Dictionary mapping subject IDs to Subject instances
        """
        study = self.studies.get(study_number)
        if study is None:
            return {}
        return study.get_from_study_by_emotion(emotions)

    def get_features(self, study_number: int) -> set[str]:
        """
        Get physiological measurement features for a specific study.
        
        Args:
            study_number: Study number (1-7)
            
        Returns:
            Set of feature names (e.g., 'ECG', 'EDA', 'TEMP')
        """
        study = self.studies.get(study_number)
        if study is None:
            return set()
        measurements = study.get_measurements()
        if measurements is None:
            return set()
        return set(measurements.keys())

    def get_all_features(self) -> list[str]:
        """Get all unique features across all 7 studies."""
        all_features = set()
        for study_number in range(1, 8):
            features = self.get_features(study_number)
            all_features.update(features)
        return sorted(all_features)

    def get_common_features(self, study_numbers: list[int]) -> set[str]:
        """
        Get features common to all specified studies.
        
        Args:
            study_numbers: List of study numbers to compare
            
        Returns:
            Set of features present in all specified studies
        """
        feature_sets = [self.get_features(num) for num in study_numbers]
        valid_sets = [fs for fs in feature_sets if fs]
        if not valid_sets:
            return set()
        return set.intersection(*valid_sets)


    def get_features_for_study(self, study_number: int) -> set[str]:
        return self.get_features(study_number)

    def get_overlap_features_across_studies(self, study_numbers: list[int]) -> set[str]:
        return self.get_common_features(study_numbers)

    def get_subject_metadata_byID(self, study_number: int, subject_id: int) -> Subject | None:
        return self.get_subject_metadata(study_number, subject_id)

    def get_stimuli_meta(self) -> pd.DataFrame:
        return self.loader.get_stimui()
