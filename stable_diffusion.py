# conda create -n sd python=3.11
# conda activate sd
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install diffusers transformers accelerate safetensors
# pip install datasets peft bitsandbytes

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
import gc


from tqdm import tqdm
import os
from peft import LoraConfig, get_peft_model, PeftModel
import bitsandbytes as bnb


# CUDA allocator
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# image transform
def transform(examples):
    transform_fn = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    pixel_values = []
    for img in examples["image"]:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        pixel_values.append(transform_fn(img))
    examples["pixel_values"] = pixel_values
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
    batch_size = 1
    num_epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    selected_genera = config["selected_genera"]
    
    # Gradient accumulation -> dealing with memory and low batch sizes
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 8)

    # model to use
    model_id = config["model_path"]

    # Define LoRA config
    # fine-tuning large models efficiently by only training small additional weights
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none"
    )

    for genus in selected_genera:
        # output for lora weights
        print(f"\n{'='*50}")
        print(f"Training on genus: {genus}")
        print(f"{'='*50}\n")
        output_dir = os.path.join(output, genus)
        lora_output_dir = os.path.join(output_dir, "lora_weights")

        train_data_dir = os.path.join(train, genus)
        os.makedirs(output_dir, exist_ok=True)

        # clear memory cache
        torch.cuda.empty_cache()
        gc.collect()

        # load data
        dataset = load_dataset("imagefolder", data_dir=train_data_dir, split="train")
        dataset = dataset.map(transform, batched=True)
        dataset = dataset.remove_columns(["image"])
        dataset.set_format(type="torch", columns=["pixel_values"])
        print(f"Dataset loaded: {len(dataset)} images")

        # load model
        print("Loading model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # diffusion pipeline with  low_cpu_mem_usage
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        # Keep components on CPU initially
        # vae - autoencorder
        vae = pipe.vae
        vae.eval()
        vae.requires_grad_(False)

        text_encoder = pipe.text_encoder
        text_encoder.eval()
        text_encoder.requires_grad_(False)

        # Get UNet and apply LoRA before moving to GPU
        # unet for facilitating diffusion
        unet = pipe.unet
        print("Applying LoRA to UNet...")
        unet = get_peft_model(unet, lora_config)

        # Clear pipeline to free memory before moving anything to GPU
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

        print("Pipeline cleared. Moving UNet to GPU...")

        # Move UNet to GPU
        unet = unet.to(device)
        unet.train()
        unet.enable_gradient_checkpointing()
        print("✓ UNet on GPU with gradient checkpointing")
        
        # 8-bit optimizer for efficiency
        optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=learning_rate)
        noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        # compute text embeddings on CPU, then move to GPU
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        prompt = f"a photo of a {genus} tree"
        with torch.no_grad():
            text_input = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=77
            ).input_ids
            encoder_hidden_states = text_encoder(text_input)[0].to(device)

        print(f"Text embeddings computed and moved to GPU")

        # clear text encoder from memory
        del text_encoder, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        # Pre-encode all images to latents to save GPU memory
        # latents - compressed, non pixel version of image
        print("Pre-encoding images to latents")
        latent_dataset = []
        temp_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        vae = vae.to(device)
        with torch.no_grad():
            for batch in tqdm(temp_dataloader, desc="Encoding images"):
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latent_dataset.append(latents.cpu())

        # Move vae back to cpu and clear it
        vae = vae.to("cpu")
        del vae
        gc.collect()
        torch.cuda.empty_cache()
        print(f"All images pre-encoded to latents, VAE cleared from GPU")

        # Create new dataset from latents
        latent_dataset = torch.cat(latent_dataset, dim=0)

        # training loop with gradient accumulation
        dataloader = DataLoader(
            torch.utils.data.TensorDataset(latent_dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        num_train_steps = 10000
        
        optimizer.zero_grad()
        
        print("Starting training...")
        for step, batch in enumerate(tqdm(dataloader, total=num_train_steps)):
            if step >= num_train_steps:
                break
            
            try:
                # Get latents
                latents = batch[0].to(device, dtype=torch.float16)

                # Add noise
                noise = torch.randn_like(latents, device=device)
                timesteps = torch.randint(
                    0, 
                    noise_scheduler.config.num_train_timesteps, 
                    (latents.shape[0],),
                    device=device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Expand encoder hidden states to match batch size
                encoder_hidden_states_batch = encoder_hidden_states.repeat(latents.shape[0], 1, 1)
                
                # Forward pass
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states_batch).sample
                
                # calculate loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
                # scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                # Update weights every gradient_accumulation_steps
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()

                if step % 100 == 0:
                    print(f"Step {step}/{num_train_steps} | Loss: {loss.item() * gradient_accumulation_steps:.4f}")
                    print("Generating sample image...")
                # intermediate image generation
                if step % 1000 == 0 and step > 0:
                    print(f"Generating preview at step {step}...")

                    checkpoint_dir = os.path.join(output_dir, f"checkpoint_{step}")
                    unet.save_pretrained(checkpoint_dir)
                    unet.eval()

                    try:
                        # load new pipeline on CPU
                        preview_pipe = StableDiffusionPipeline.from_pretrained(
                            model_id,
                            torch_dtype=torch.float16,
                            safety_checker=None,
                            low_cpu_mem_usage=True
                        )

                        # apply LoRA weights
                        preview_unet = PeftModel.from_pretrained(preview_pipe.unet, checkpoint_dir)
                        preview_pipe.unet = preview_unet
                        preview_pipe = preview_pipe.to("cpu")

                        with torch.no_grad():
                            preview_image = preview_pipe(
                                f"a photo of a {genus} tree",
                                guidance_scale=7.5,
                                num_inference_steps=30,
                                height=256,
                                width=256
                            ).images[0]

                        preview_path = os.path.join(checkpoint_dir, f"preview_step_{step}.png")
                        preview_image.save(preview_path)
                        print(f"Preview saved: {preview_path}")

                        del preview_pipe, preview_unet
                        gc.collect()

                    except Exception as e:
                        print(f"Preview failed: {e}")

                    torch.cuda.empty_cache()
                    unet.train()
                
                # clear intermediate variables
                del latents, noise, noisy_latents, noise_pred, loss
                if step % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nOOM at step {step}. Clearing cache and continuing...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e

        # Final optimizer step if remaining gradients
        optimizer.step()
        optimizer.zero_grad()

        # save lora weigths
        print("Saving LoRA weights...")
        unet.save_pretrained(lora_output_dir)
        
        # save config
        config_save = {
            "base_model": model_id,
            "lora_rank": 8,
            "genus": genus,
            "training_steps": num_train_steps
        }
        with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
            json.dump(config_save, f, indent=2)
        
        print(f"Model saved to {lora_output_dir}")

        # clear memory before generation
        del unet, optimizer
        gc.collect()
        torch.cuda.empty_cache()

        # final image generation with LoRA
        print("Generating sample image...")
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None
            )
            
            # apply LoRA to unet
            base_unet = pipe.unet
            base_unet = PeftModel.from_pretrained(base_unet, lora_output_dir)
            
            pipe.unet = base_unet
            pipe.to(device)
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()

            prompts = [
                f"a photo of a {genus} tree",
                f"a {genus} tree in a forest",
                f"close up photo of {genus} tree bark and leaves",
                f"a tall {genus} tree",
                f"{genus} tree in nature"
            ]
            for idx, genPrompt in enumerate(prompts):
                with torch.no_grad():
                    image = pipe(
                        genPrompt, 
                        guidance_scale=7.5, 
                        num_inference_steps=30,
                        height=256,
                        width=256
                    ).images[0]
                
                save_path = os.path.join(output_dir, f"generated_tree{idx}.png")
                image.save(save_path)
                print(f"Generated image saved to {save_path}")
            
        except Exception as e:
            print(f"Image generation failed: {e}")
            print("LoRA weights saved successfully. You can generate images later.")
        
        # cleanup for next genus
        try:
            del pipe
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"Completed training for {genus}\n")


if __name__ == "__main__":
    main()