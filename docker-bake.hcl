# ============================================
# Docker Bake Configuration for Unsloth Worker
# ============================================

variable "DOCKERHUB_REPO" {
  default = "runpod"
}

# Rename image (was worker-v1-vllm → now worker-v1-unsloth)
variable "DOCKERHUB_IMG" {
  default = "worker-v1-unsloth"
}

variable "RELEASE_VERSION" {
  default = "latest"
}

# Optional HF token for private model pulls
variable "HUGGINGFACE_ACCESS_TOKEN" {
  default = ""
}

group "default" {
  targets = ["worker-unsloth"]
}

target "worker-unsloth" {
  tags       = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}"]
  context    = "."
  dockerfile = "Dockerfile"
  platforms  = ["linux/amd64"]
}
