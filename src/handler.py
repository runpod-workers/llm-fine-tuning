import runpod
import os
from train import train
from utils import get_output_dir
from huggingface_hub._login import login
import yaml

BASE_VOLUME = os.environ.get("BASE_VOLUME", "/runpod-volume")
if not os.path.exists(BASE_VOLUME):
    os.makedirs(BASE_VOLUME)

# axolotl reads its settings from this file
CONFIG_PATH = "/workspace/test_config.yaml"
# job input credential key -> environment variable read by axolotl / wandb / hf
CREDENTIAL_ENV = {"wandb_api_key": "WANDB_API_KEY", "hf_token": "HF_TOKEN"}

logger = runpod.RunPodLogger()


async def handler(job):
    """Fine-tune with axolotl.

    Credentials are optional: `hf_token` is needed only for gated models or
    datasets and for pushing to the Hub, `wandb_api_key` only when the args
    set `wandb_project`. Raises if preprocessing or training fails.
    Returns the run id and the output directory on the volume.
    """
    runpod_job_id = job["id"]
    inputs = job["input"]
    run_id = inputs["run_id"]
    args = inputs["args"]

    # Set output directory
    output_dir = os.path.join(BASE_VOLUME, get_output_dir(run_id))
    args["output_dir"] = output_dir

    # Add run_name and job_id to args before saving
    args["run_name"] = run_id
    args["runpod_job_id"] = runpod_job_id

    yaml_data = yaml.dump(args, default_flow_style=False)
    with open(CONFIG_PATH, "w") as file:
        file.write(yaml_data)

    credentials = inputs.get("credentials") or {}
    set_env = {
        env: credentials[key] for key, env in CREDENTIAL_ENV.items() if credentials.get(key)
    }
    os.environ.update(set_env)

    try:
        if credentials.get("hf_token"):
            login(token=credentials["hf_token"])

        logger.info("Starting Training.")
        async for result in train(CONFIG_PATH):
            logger.info(result)
        logger.info("Training Complete.")
    finally:
        for env in set_env:
            os.environ.pop(env, None)

    return {"run_id": run_id, "output_dir": output_dir}


runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True
})
