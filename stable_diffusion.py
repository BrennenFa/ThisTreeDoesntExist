# conda create -n sd python=3.11
# conda activate sd
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install diffusers transformers accelerate safetensors
# pip install datasets




# stable diffusion

from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5" 

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "Recreate this tree genera, realistic, 512x512"
image = pipe(prompt).images[0]
image.save("plant.png")
