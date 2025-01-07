import os 
import yaml 



def verify_env(logger, rp_job_id):
    vars =  ["WANDB_API_KEY", "HF_TOKEN"]
    
    for key in vars:
        if key not in os.environ:
            logger.error(
                f"Enviornment variable {key} not found. Please set it before running the script.", job_id=rp_job_id
            )
            raise ValueError(f"Enviornment variable {key} not found. Please set it before running the script.")


def get_output_path(run_id):
    path = f"fine-tuning/{run_id}"
    return path



# def make_valid_config(input_args):
#     """
#     Currently limited by all possible axolotl args, no defaults
#     :param input_args: dict of input args
#     """
#     all_args = yaml.safe_load(open("config/config.yaml", "r"))
#     if not input_args:
#         print("No args provided, using defaults")
#     else:
#         all_args.update(input_args)
#     return all_args



def make_valid_config(input_args):
    """
    Creates and saves updated config file, returns the path to the new config
    :param input_args: dict of input args
    :return: str, path to the updated config file
    """
    # Load default config
    all_args = yaml.safe_load(open("config/config.yaml", "r"))
    
    if not input_args:
        print("No args provided, using defaults")
    else:
        all_args.update(input_args)
    
    # Create updated config path
    updated_config_path = "config/updated_config.yaml"
    
    # Save updated config to new file
    with open(updated_config_path, "w") as f:
        yaml.dump(all_args, f)
    
    return updated_config_path