"""
POPANE Emotion - Package for emotion study data analysis

Main API:
    POPANE: Main facade class for accessing all functionality
    
Core Components:
    - studies: Study1-7 classes for each emotion study
    - dataloader: POPANEDataLoader for raw data access
    - preprocessing: Data transformation and cleaning
    - models: Machine learning models for emotion prediction
    - visualization: Plotting and figure generation
    - db: Database management and optimization
"""
from emotion.core.popane import POPANE
from emotion.dataloader import POPANEDataLoader
from emotion.db.db import POPANEDB
from emotion.studies.study import Study, get_study
from emotion.studies.subject import Subject, POPANEMetadata
from emotion.utils.utils import get_size
from emotion.visualization.figure import POPANEFigureGenerator

__version__ = "0.1.0"

# Main API exports
__all__ = [
    # Main facade
    "POPANE",

    # Studies
    "Study",
    "get_study",
    "Subject",
    "POPANEMetadata",

    # Data access
    "POPANEDataLoader",
    "POPANEDB",

    # Visualization
    "POPANEFigureGenerator",

    # Utils
    "get_size",

    # Version
    "__version__", ]
