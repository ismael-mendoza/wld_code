import os
from pathlib import Path

# names to be used.
root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = root_dir.joinpath('data')
logs_dir = root_dir.joinpath('logs')
params_dir = data_dir.joinpath('params')
