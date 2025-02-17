from huggingface_hub import hf_hub_download
import os

os.makedirs("zonos-v0.1", exist_ok=True)
config_path = hf_hub_download(repo_id="Zyphra/Zonos-v0.1-transformer", filename="config.json", local_dir="zonos-v0.1")
model_path = hf_hub_download(repo_id="Zyphra/Zonos-v0.1-transformer", filename="model.safetensors", local_dir="zonos-v0.1")
print(config_path)
print(model_path)
