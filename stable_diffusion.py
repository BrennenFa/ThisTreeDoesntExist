# conda create -n sd python=3.11
# conda activate sd
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install diffusers transformers accelerate safetensors
# pip install datasets




# stable diffusion
import torch
from datasets import load_dataset
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers import DDPMScheduler, AutoencoderKL
from diffusers import StableDiffusionPipeline
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import json
import argparse

from tqdm import tqdm
import os



# pixel to tensors
def transform(examples):
    examples["pixel_values"] = [torch.tensor(np.array(img)).permute(2, 0, 1) / 255.0 for img in examples["image"]]
    return examples





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # model config
    experiment_name = config["experiment"]
    output_dir = os.path.join(config["output_path"], f"sd_finetuned_{experiment_name}")
    train_data_dir = config["train_data_dir"]
    batch_size = config["batch_size"]
    num_epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    selected_genera = config["selected_genera"]

    # model to use
    model_id = "runwayml/stable-diffusion-v1-5"

    os.makedirs(output_dir, exist_ok=True)

    # load data
    dataset = load_dataset("imagefolder", data_dir=train_data_dir, split="train")
    print("data loaded")


    # load model
    print("load model")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to("cuda")

    unet = pipe.unet
    optimizer = optim.AdamW(unet.parameters(), lr=learning_rate)
    accelerator = Accelerator(mixed_precision="fp16")

    pipe.unet, optimizer = accelerator.prepare(unet, optimizer)

    # training loop (simple fine-tune)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    num_train_steps = 500
    for step, batch in enumerate(tqdm(dataloader, total=num_train_steps)):
        if step >= num_train_steps:
            break
        # Random noise
        noise = torch.randn_like(batch["pixel_values"][0].unsqueeze(0)).to("cuda")
        noisy_image = batch["pixel_values"][0].unsqueeze(0).to("cuda") + 0.1 * noise
        output = unet(noisy_image, 0.5).sample
        loss = (output - batch["pixel_values"][0].unsqueeze(0).to("cuda")).abs().mean()

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()


        if step % 50 == 0:
            print(f"Step {step}/{num_train_steps} | Loss: {loss.item():.4f}")

    # save model
    pipe.save_pretrained(output_dir)

    # image geneartion
    print("generating image")
    pipe = StableDiffusionPipeline.from_pretrained(
        output_dir,
        torch_dtype=torch.float16
    ).to("cuda")

    prompt = "Generate a tree of this genera"
    image = pipe(prompt, guidance_scale=7.5, num_inference_steps=30).images[0]
    save_path = os.path.join(output_dir, "generated_tree.png")
    image.save(save_path)

if __name__ == "__main__":
    main()
