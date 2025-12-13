"""
This is the main script to test the data loader and processor.
"""

from time import sleep

from test.test_gzip import data1
from emotion.dataloader import POPANEDataLoader, PopaneDataLoader
from emotion.models import EmotionRandomForest
from emotion.preprocessing import ECG_SmoothTransformer
from emotion.studies import Subject, Study1 
from emotion import POPANEFigureGenerator
data_loader = POPANEDataLoader()


def main():

    loader = POPANEDataLoader()
    print(loader.get_study_metadata(1))

if __name__ == "__main__":
    main()