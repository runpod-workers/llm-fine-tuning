import argparse
from typing import Any, Optional
import os
import json
import yaml
from axolotl.utils.config.models.input.v0_4_1 import AxolotlInputConfig

"""
Example:

[ENV VARS]
AXOLOTL_BASE_MODEL=TinyLlama/TinyLlama_v1.1
AXOLOTL_DATASETS='[{"path":"mhenrichsen/alpaca_2k_test","type":"alpaca"}]'
AXOLOTL_OUTPUT_DIR=./outputs/my_training

[Usage]
config = load_config_with_overrides("config_template.yml")
save_config(config, "config.yml")
"""

DEFAULT_PREFIX = "AXOLOTL_"


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
    config_path: str, env_prefix: str = DEFAULT_PREFIX
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
    if not config_path.startswith("/"):
        # absolute path
        config_path = os.path.join(os.path.dirname(__file__), config_path)

    with open(config_path, "r") as f:
        print(f"🛠️ Generating from template: {config_path}")
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
    config_dict = config.model_dump(mode="json", exclude_none=True)

    if not output_path.startswith("/"):
        # absolute path
        output_path = os.path.join(os.path.dirname(__file__), output_path)

    # Ensure output directory exists
    if output_dir := os.path.dirname(output_path):
        os.makedirs(output_dir, exist_ok=True)

    # Save to YAML
    with open(output_path, "w") as f:
        yaml.safe_dump(config_dict, f, sort_keys=True, default_flow_style=False)

    print(f"💾 Saved configuration to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an Axolotl training configuration based on the template and environment variables."
    )
    parser.add_argument(
        "--template", type=str, required=True, help="Path to the template YAML file."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the output YAML file."
    )

    if len(os.sys.argv) == 1:
        parser.print_help()
        os.sys.exit(1)

    args = parser.parse_args()

    try:
        config = load_config_with_overrides(args.template)
        save_config(config, args.output)

    except Exception as e:
        print(f"❌ Error processing configuration: {str(e)}")
        raise
