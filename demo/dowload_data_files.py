"""
This script is used to download data files for the POPANE dataset.
"""

from emotion import POPANEDataLoader
from emotion.dataloader.downloader import POPANEDownloader

def main():
    from emotion import POPANEDB
    # Download the data files
    # Can select which dataset to download by passing arguments
    # POPANEDataLoader().download_data(studies=["meta", 1, 2]) // Downloads only meta, 1, and 2 studies
    # POPANEDataLoader().download_data(studies=["meta"]) // Downloads only meta study
    # POPANEDataLoader().download_data(studies=["meta", 1, 2, 3]) // Downloads meta, 1, 2, and 3 studies
    popane = POPANEDataLoader()
    popane.download_data()
    popane.unzip_files()
    # Create the database
    popane_db = POPANEDB()
    popane_db.createDB()


if __name__ == "__main__":
    main()