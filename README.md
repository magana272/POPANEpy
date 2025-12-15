# POPANEpy

A comprehensive Python package for analyzing physiological signals and emotion classification from the POPANE (Physiological Observation of Person Affect and Negative Emotions) dataset.

## Overview

POPANEpy provides a unified interface for:
- Loading and accessing emotion study data from 7 different studies
- Preprocessing physiological signals (ECG, EDA, blood pressure, respiration, temperature)
- Feature extraction and analysis
- Emotion classification using machine learning
- Visualization and statistical analysis

## Features

- **Unified API**: POPANE facade class provides simple access to all studies
- **Database Integration**: DuckDB-based storage for efficient querying
- **Preprocessing**: Signal filtering, cleaning, and transformation utilities
- **Machine Learning**: Random Forest classifier for emotion prediction
- **Visualization**: Comprehensive plotting tools for signals and analysis results

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/POPANEpy.git
cd POPANEpy

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from emotion import POPANE

# Initialize the POPANE interface
popane = POPANE()

# Get unique emotions from Study 1
emotions = popane.get_unique_emotions(1)

# Get subject data
subject = popane.get_subject(study_number=1, subject_id=123)

# Get common features across studies
common_features = popane.get_common_features([1, 2, 3])
```

## Analysis Results


## Machine Learning Results

### Model Performance


### Confusion Matrix


The confusion matrix shows strong performance in distinguishing between emotion categories.

### Feature Importance

## Project Structure

```
POPANEpy/
├── src/emotion/
│   ├── core/           # POPANE facade and main API
│   ├── studies/        # Study-specific classes
│   ├── dataloader/     # Data loading utilities
│   ├── preprocessing/  # Signal processing and cleaning
│   ├── models/         # Machine learning models
│   ├── visualization/  # Plotting utilities
│   ├── db/            # Database management
│   └── utils/         # Helper functions
├── data/
│   ├── raw/           # Raw study data
│   └── processed/     # Processed database
├── notebooks/         # Jupyter notebooks for analysis
├── test/             # Unit tests
└── figures/          # Generated visualizations
```

## Testing

The project includes comprehensive unit tests with 130 tests covering all major components:

```bash
# Run all tests
python -m unittest discover test/

# Run with coverage
coverage run -m unittest discover test/
coverage report
coverage html
```

Current test coverage: **48%**

## Documentation

- **ARCHITECTURE.md**: Detailed package structure and design decisions
- **notebooks/analysis.ipynb**: Complete analysis workflow with visualizations
- API documentation: See docstrings in source code

## Usage Examples

### Loading Data from Database

```python
import duckdb
conn = duckdb.connect("data/processed/propane_emotion.db")
df = conn.execute("SELECT * FROM study1 LIMIT 1000").fetchdf()
```

### Training a Classifier

```python
from emotion.models.random_forest import EmotionRandomForest

model = EmotionRandomForest(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Preprocessing Signals

```python
from emotion.preprocessing.dataprocess import clean_DBP

cleaned_data = clean_DBP(subject_data)
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Citation

If you use POPANEpy in your research, please cite:

```
@software{popanepy2024,
  title={POPANEpy: A Python Package for Emotion Analysis from Physiological Signals},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/POPANEpy}
}
```

## Acknowledgments

- POPANE dataset providers
- Contributors and maintainers
- Scientific Python community (NumPy, Pandas, Scikit-learn, Matplotlib)

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
