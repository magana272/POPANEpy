"""
This an example script to create a database for the POPANE dataset.
"""
from emotion import POPANEDB
import sys
def main():
    POPANEDB().createDB()

if __name__ == "__main__":
    main()



