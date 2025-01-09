variable "PUSH" {
  default = "true"
}

variable "REPOSITORY" {
  default = "runpod"
}

variable "BASE_IMAGE_VERSION" {
  default = "dev"
}

group "all" {
  targets = ["main"]
}


group "main" {
  targets = ["worker"]
}

 
target "worker" {
  tags = ["${REPOSITORY}/llm-finetuning:${BASE_IMAGE_VERSION}"]
  context = "."
  dockerfile = "Dockerfile"
  args = {
    BASE_IMAGE_VERSION = "${BASE_IMAGE_VERSION}"
    WORKER_CUDA_VERSION = "12.1.0"
  }
  output = ["type=docker,push=${PUSH}"]
}