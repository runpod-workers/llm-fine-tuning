

<h1>LLM Training- Full finetune, LoRA, QLoRa etc. Llama/Mistral/Gemma</h1>


### Docker Dev Image: 
Please use this image for Dev/Testing purpose.

`runpod/llm-finetuning:dev`

### Sample Request: 

json 

```
{
  "input": {
    "user_id": "user",
    "model_id": "llama-test",
    "run_id": "test-run",
    "credentials": {
      "wandb_api_key": "",
      "hf_token": "<Use your HF_TOKEN>"
    },
    "args": {
      "base_model": "NousResearch/Llama-3.2-1B",
      "load_in_8bit": false,
      "load_in_4bit": false,
      "strict": false,
      "datasets": [
        {
          "path": "teknium/GPT4-LLM-Cleaned",
          "type": "alpaca"
        }
      ],
      "dataset_prepared_path": "last_run_prepared",
      "val_set_size": 0.1,
      "output_dir": "./outputs/lora-out",
      "adapter": "lora",
      "sequence_len": 2048,
      "sample_packing": true,
      "eval_sample_packing": true,
      "pad_to_sequence_len": true,
      "lora_r": 16,
      "lora_alpha": 32,
      "lora_dropout": 0.05,
      "lora_target_modules": [
        "gate_proj",
        "down_proj",
        "up_proj",
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj"
      ],
      "gradient_accumulation_steps": 2,
      "micro_batch_size": 2,
      "num_epochs": 1,
      "optimizer": "adamw_8bit",
      "lr_scheduler": "cosine",
      "learning_rate": 0.0002,
      "train_on_inputs": false,
      "group_by_length": false,
      "bf16": "auto",
      "tf32": false,
      "gradient_checkpointing": true,
      "logging_steps": 1,
      "flash_attention": true,
      "loss_watchdog_threshold": 5,
      "loss_watchdog_patience": 3,
      "warmup_steps": 10,
      "evals_per_epoch": 4,
      "saves_per_epoch": 1,
      "weight_decay": 0,
      "special_tokens": {
        "pad_token": "<|end_of_text|>"
      }
    }
  }
}

### Errors: 
- if you face any issues with the Flash Attention-2, Delete yoor worker and Re-start.




