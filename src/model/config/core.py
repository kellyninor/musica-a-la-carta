from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from strictyaml import YAML, load

#Project Directories
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained"


class AppConfig(BaseModel):
    """
    Application-level config.
    """
    package_name: str
    data_file: str
    pipeline_save_file: str

class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """    
    atributos_deseados: List[str]
    scaling_method: str
    generos_usuario: List[str]   
    sentimiento_usuario: str
    n_components: int
    top_n: int

class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig
    

def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""
    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")

def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    app_config = AppConfig(**parsed_config.data)
    model_config = ModelConfig(**parsed_config.data)

    #_config = Config(
    #     app_config=AppConfig(**parsed_config.data),
    #     model_config=ModelConfig(**parsed_config.data),
    # )

    return app_config, model_config

app_config, model_config = create_and_validate_config()


