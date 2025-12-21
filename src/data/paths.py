
import os
import yaml
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


PATHS_FILE = REPO_ROOT / 'paths.yaml'
if not PATHS_FILE.exists():
    print("Warning: paths.yaml not found. Using paths.example.yaml")
    print("Please copy paths.example.yaml to paths.yaml and update with your local paths.")
    PATHS_FILE = REPO_ROOT / 'paths.example.yaml'

with open(PATHS_FILE, 'r') as f:
    config = yaml.safe_load(f)

PROJECT_ROOT = Path(config['project_root'])


RAW_DATA_PATH = Path(config.get('raw_data_path', PROJECT_ROOT / 'data' / 'raw'))
FEATURES_PATH = Path(config.get('features_path', PROJECT_ROOT / 'data' / 'features'))

# "artifacts" is our canonical features/output folder for generated datasets
ARTIFACTS_PATH = FEATURES_PATH

# Standard artifact file names produced by scripts/00_preprocessing.py
X_TRAIN_PATH = ARTIFACTS_PATH / 'X_train.npy'
Y_TRAIN_PATH = ARTIFACTS_PATH / 'y_train.npy'
X_TEST_PATH = ARTIFACTS_PATH / 'X_test.npy'
Y_TEST_PATH = ARTIFACTS_PATH / 'y_test.npy'
MANIFEST_PATH = ARTIFACTS_PATH / 'manifest.json'

RESULTS_PATH = REPO_ROOT / 'results'
FIGURES_PATH = REPO_ROOT / 'figures'
LOGS_PATH = REPO_ROOT / 'logs'

# Ensure all directories exist
RESULTS_PATH.mkdir(exist_ok=True)
FIGURES_PATH.mkdir(exist_ok=True)
LOGS_PATH.mkdir(exist_ok=True)
FEATURES_PATH.mkdir(parents=True, exist_ok=True)
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)


PATHS = {
    'project_root': PROJECT_ROOT,
    'raw_data': RAW_DATA_PATH,
    'features': FEATURES_PATH,
    'artifacts': ARTIFACTS_PATH,
    # Preprocessed dataset artifacts (memmap .npy files + manifest)
    'X_train': X_TRAIN_PATH,
    'y_train': Y_TRAIN_PATH,
    'X_test': X_TEST_PATH,
    'y_test': Y_TEST_PATH,
    'manifest': MANIFEST_PATH,
    'results': RESULTS_PATH,
    'figures': FIGURES_PATH,
    'logs': LOGS_PATH,
}
