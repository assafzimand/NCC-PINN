"""IO utilities for configuration and directory management."""

from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import yaml


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        path: Path to the config YAML file (default: "config/config.yaml")

    Returns:
        Dictionary containing all configuration parameters
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def make_run_dir(problem: str, layers: list, act: str) -> Path:
    """
    Create a run directory with standardized naming and timestamp.

    Args:
        problem: Problem name (e.g., "schrodinger")
        layers: List of layer sizes (e.g., [2, 50, 100, 50, 2])
        act: Activation function name (e.g., "tanh")

    Returns:
        Path object to the created run directory
        Structure: outputs/<problem>_layers-<...>_act-<act>/<timestamp>/
    """
    # Format layers as string: "2-50-100-50-2"
    layers_str = "-".join(map(str, layers))

    # Create architecture directory name: <problem>-<...>-<activation>
    arch_dir_name = f"{problem}-{layers_str}-{act}"
    
    # Create timestamp for unique run: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the full path: outputs/<architecture>/<timestamp>/
    arch_dir = Path("outputs") / arch_dir_name
    run_dir = arch_dir / timestamp

    # Create the directory and subdirectories
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_plots").mkdir(exist_ok=True)
    (run_dir / "ncc_plots").mkdir(exist_ok=True)

    return run_dir
