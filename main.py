"""
This is the main script to test the data loader and processor.
"""
from dataloader.popaneloader import POPANEMETADataLoader
# from emotion.models import EmotionRandomForest
# from emotion.preprocessing import ECG_SmoothTransformer
from emotion.studies import Subject
# from emotion import POPANEFigureGenerator
from emotion import utils
from emotion import POPANE
def main():
    # TODO: DEMONSTRATE POPANE APPLICATION
    #  1) POPANE().get_study_metadata(study_number: int)->POPANEMetaDatA | None
    #  2) POPANE().get_subject_ids(self, study_number) -> list[int]:
    #  3) POPANE().
    #  4) POPANE().
    #  5) POPANE().
    #  6) POPANE().
    #  7) POPANE().
    #  8) POPANE().
    # print(POPANE().get_study_metadata(1))
    # print(POPANE().get_study_metadata(2))
    # print(POPANE().get_study_metadata(3))
    # print(POPANE().get_study_metadata(4))
    # print(POPANE().get_study_metadata(5))
    # p = POPANE()
    # s = p.get_stimuli_meta()
    # print(s.head())
    print(utils.get_size())
if __name__ == "__main__":
    main()