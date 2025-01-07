import runpod
import os
from train import train
from utils import make_valid_config, get_output_dir, validate_env
from huggingface_hub._login import login

BASE_VOLUME = os.environ.get("BASE_VOLUME", "/runpod-volume")
if not os.path.exists(BASE_VOLUME):
    os.makedirs(BASE_VOLUME)

logger = runpod.RunPodLogger()


async def handler(job):
    runpod_job_id = job["id"]
    inputs = job["input"]
    run_id = inputs["run_id"]

    args = inputs["args"]
    output_dir = os.path.join(BASE_VOLUME, get_output_dir(run_id))
    args["output_dir"] = output_dir
    args = make_valid_config(args)
    args["run_name"] = run_id
    args["runpod_job_id"] = runpod_job_id

    credentials = inputs["credentials"]
    os.environ["WANDB_API_KEY"] = credentials["wandb_api_key"]
    os.environ["HF_TOKEN"] = credentials["hf_token"]

    validate_env(logger, runpod_job_id)
    login(token=os.environ["HUGGING_FACE_HUB_TOKEN"])

    logger.info(f"Starting Training.", job_id=runpod_job_id)
    async for result in train(args):
        logger.info(result, job_id=runpod_job_id)
    logger.info(f"Training Complete.", job_id=runpod_job_id)
    del os.environ["WANDB_API_KEY"]
    del os.environ["Hf_TOKEN"]


runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True
})