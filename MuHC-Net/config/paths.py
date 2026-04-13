from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directory (put your CSV files here)
DATA_DIR = PROJECT_ROOT / 'data'

# Output directory (models, features, results)
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)