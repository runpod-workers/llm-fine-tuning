import runpod
import os
from train import train
from utils import make_valid_config, get_output_dir, validate_env, set_config_env_vars
from huggingface_hub._login import login
import yaml

BASE_VOLUME = os.environ.get("BASE_VOLUME", "/runpod-volume")
if not os.path.exists(BASE_VOLUME):
    os.makedirs(BASE_VOLUME)

logger = runpod.RunPodLogger()


async def handler(job):
    runpod_job_id = job["id"]
    inputs = job["input"]
    run_id = inputs["run_id"]
    args = inputs["args"]
    
    # set_config_env_vars(args)
    
    # Set output directory
    output_dir = os.path.join(BASE_VOLUME, get_output_dir(run_id))
    args["output_dir"] = output_dir
    
    # First save args to a temporary config file
    config_path = "/workspace/test_config.yaml"
    
        
    # Add run_name and job_id to args before saving
    args["run_name"] = run_id
    args["runpod_job_id"] = runpod_job_id

    yaml_data = yaml.dump(args, default_flow_style=False)
    with open(config_path, "w") as file:
        file.write(yaml_data)
    
    # Handle credentials
    credentials = inputs["credentials"]
    os.environ["WANDB_API_KEY"] = credentials["wandb_api_key"]
    os.environ["HF_TOKEN"] = credentials["hf_token"]
    
    validate_env(logger, runpod_job_id)
    login(token=os.environ["HF_TOKEN"])
    
    logger.info(f"Starting Training.")
    async for result in train(config_path):  # Pass the config path instead of args
        logger.info(result)
    logger.info(f"Training Complete.")
    
    # Cleanup
    del os.environ["WANDB_API_KEY"]
    del os.environ["HF_TOKEN"]



runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True
})