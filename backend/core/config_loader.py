import os
from omegaconf import OmegaConf, DictConfig

_config: DictConfig = None

def load_config(config_file: str = "../../configs/app_config.yaml") -> DictConfig:
    """
    Loads the configuration from a YAML file using OmegaConf.
    Uses a cached value to avoid reloading on repeated calls.

    Args:
        config_file (str): Relative path to the YAML config file.

    Returns:
        DictConfig: Parsed configuration object.
    """
    global _config

    if _config is None:
        try:
            base_dir = os.path.dirname(__file__)
            config_path = os.path.abspath(os.path.join(base_dir, config_file))
            _config = OmegaConf.load(config_path)
            print(f"[INFO] Configuration loaded from {config_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load configuration: {e}")
            raise

    return _config