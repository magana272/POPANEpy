"""
This script is used to download data files for the POPANE dataset.
"""

from emotion import POPANEDataLoader
def main():
    from emotion import POPANEDB
    # Download the data files
    # Can select which dataset to download by passing arguments
    # POPANEDataLoader().download_data(studies=["meta", 1, 2]) // Downloads only meta, 1, and 2 studies
    # POPANEDataLoader().download_data(studies=["meta"]) // Downloads only meta study
    # POPANEDataLoader().download_data(studies=["meta", 1, 2, 3]) // Downloads meta, 1, 2, and 3 studies
    POPANEDataLoader().download_data() # Default: Downloads all studies
    # Create the database

if __name__ == "__main__":
    main()