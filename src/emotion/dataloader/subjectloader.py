from emotion.studies.study_config import StudyConfig
from emotion.studies.subject import Subject


class SubjectLoader:
    def __init__(self, config: StudyConfig) -> None:
        self.config = config

    def get_subjects(self) -> list[Subject]:
        pass

    def get_subject(self, subject_id: int) -> Subject:
        pass
