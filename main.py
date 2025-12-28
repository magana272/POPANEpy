"""
This is the main script to test the data loader and processor.
"""
from emotion.dataloader.popaneloader import POPANEDataLoader
# from emotion.models import EmotionRandomForest
# from emotion.preprocessing import ECG_SmoothTransformer
from emotion.studies import Subject
# from emotion import POPANEFigureGenerator
from emotion import utils
from emotion import POPANEDB
def main():
    from emotion import POPANE
    # Initialize POPANE facade
    popane = POPANE(data_dir="data/raw/")

    # Example: Get unique emotions from Study 1
    emotions_study1 = popane.get_unique_emotions(1)
    print(f"Unique emotions in Study 1: {emotions_study1}")

    # Example: Get all unique emotions across all studies
    all_emotions = popane.get_unique_emotions_for_all_studies()
    print(f"All unique emotions across all studies: {all_emotions}")

    # Example: Get subject IDs from Study 2
    subject_ids_study2 = popane.get_subject_ids(2)
    print(f"Subject IDs in Study 2: {subject_ids_study2}")

    # Example: Get a specific subject from Study 3
    subject = popane.get_subject(3, subject_id=101)
    if subject:
        print(f"Subject 101 in Study 3 metadata: {subject.metadata}")
    else:
        print("Subject 101 not found in Study 3.")

if __name__ == "__main__":
    main()