# conda create -n sd python=3.11
# conda activate sd
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install diffusers transformers accelerate safetensors
# pip install datasets




# stable diffusion

from torchvision import transforms
from PIL import Image

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



# image transform
def transform(examples):
    transform_fn = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    images = []
    for img in examples["image"]:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        images.append(transform_fn(img))
    
    examples["pixel_values"] = images
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
    output = os.path.join(config["output_path"], f"sd_finetuned_{experiment_name}")
    train = config["train_data_dir"]
    batch_size = config["batch_size"]
    num_epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    selected_genera = config["selected_genera"]

    # model to use
    model_id = "runwayml/stable-diffusion-v1-5"
    for genus in selected_genera:
        print(f"\n{'='*50}")
        print(f"Training on genus: {genus}")
        print(f"{'='*50}\n")
        output_dir = output + genus
        train_data_dir = train + genus
        os.makedirs(output_dir, exist_ok=True)

        # load data
        dataset = load_dataset("imagefolder", data_dir=train_data_dir, split="train")
        dataset = dataset.map(transform)
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


        vae = pipe.vae
        noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        # training loop (simple fine-tune)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        num_train_steps = 500
        for step, batch in enumerate(tqdm(dataloader, total=num_train_steps)):
            if step >= num_train_steps:
                break
            # Random noise

            pixel_values = torch.stack(batch["pixel_values"]).to("cuda", dtype=torch.float16)
        
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device="cuda").long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # test - text embeddings
            with torch.no_grad():
                encoder_hidden_states = pipe.text_encoder(
                    pipe.tokenizer("", return_tensors="pt", padding="max_length", max_length=77).input_ids.to("cuda")
                )[0]
                encoder_hidden_states = encoder_hidden_states.repeat(latents.shape[0], 1, 1)
            
            # noise predicitons
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
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
