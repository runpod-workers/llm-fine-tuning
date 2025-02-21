from typing import Any, Optional
import os
import json
import yaml
from axolotl.utils.config.models.input.v0_4_1 import AxolotlInputConfig

"""
Example:

[ENV VARS]
AXOLOTL_BASE_MODEL = "NousResearch/Llama-3.2-1B"
AXOLOTL_LOAD_IN_8BIT = false
AXOLOTL_LOAD_IN_4BIT = false
AXOLOTL_STRICT = false
AXOLOTL_DATASETS = '[{"path": "teknium/GPT4-LLM-Cleaned","type": "alpaca"}]'
AXOLOTL_LORA_TARGET_MODULES = '["gate_proj","down_proj","up_proj","q_proj","v_proj","k_proj","o_proj"]'
AXOLOTL_HUB_MODEL_ID = "runpod/llama-fr-lora"
AXOLOTL_SPECIAL_TOKENS = '{"pad_token": "<|end_of_text|>"}'

[Usage]
```
try:
    config = load_config_with_overrides("template_minimum.yml")
    save_config(config, "config.yml")

except Exception as e:
    print(f"Error processing configuration: {str(e)}")
    raise
```
"""

CONFIG_TEMPLATE_PATH = "config/template_minimum.yml"
DEFAULT_CONFIG_PATH = "config/default_config.yml"


def parse_env_value(value: str) -> Any:
    """Parse a string value that could be JSON into appropriate Python type."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def get_env_override(key: str, prefix: str = "") -> Optional[Any]:
    """
    Get environment variable override for a config key.
    Handles JSON structures for nested configs.
    """
    env_key = f"{prefix}{key.upper()}"
    value = os.environ.get(env_key)

    if value is None:
        return None

    return parse_env_value(value)


def load_config_with_overrides(
    config_path: str, env_prefix: str = "AXOLOTL_"
) -> AxolotlInputConfig:
    """
    Load and parse the YAML config file, applying any environment variable overrides.
    Uses the Pydantic AxolotlInputConfig for validation and parsing.

    Args:
        config_path: Path to the YAML config file
        env_prefix: Prefix for environment variables to override config values

    Returns:
        AxolotlInputConfig object with merged configuration
    """
    # Load base config from YAML
    with open(os.path.join(os.path.dirname(__file__), config_path), "r") as f:
        config_dict = yaml.safe_load(f)

    # Get all fields from the Pydantic model
    model_fields = AxolotlInputConfig.model_fields

    # Apply environment overrides
    for field_name in model_fields:
        if env_value := get_env_override(field_name, env_prefix):
            config_dict[field_name] = env_value

    # Create and validate the config
    return AxolotlInputConfig.model_validate(config_dict)


def save_config(config: AxolotlInputConfig, output_path: str) -> None:
    """
    Save the configuration to a YAML file.
    """
    # Convert to dict and remove null values
    config_dict = config.model_dump(exclude_none=True)

    # Determine absolute file path
    output_path = os.path.join(os.path.dirname(__file__), output_path)

    # Ensure output directory exists
    if output_dir := os.path.dirname(output_path):
        os.makedirs(output_dir, exist_ok=True)

    # Save to YAML
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, sort_keys=True, default_flow_style=False)

    print(f"✅  Configuration saved to: {output_path}")


if __name__ == "__main__":
    try:
        config = load_config_with_overrides(CONFIG_TEMPLATE_PATH)
        save_config(config, DEFAULT_CONFIG_PATH)

    except Exception as e:
        print(f"❌  Error processing configuration: {str(e)}")
        raise
