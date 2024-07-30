import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from samplers.ddim import DDIMSampler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Replace the scheduler with the custom DDIM sampler
# Replace the scheduler with the custom DDIM sampler
beta_start = 0.0001
beta_end = 0.02

for T in [5, 10, 50, 100]:
    # Instantiate the custom DDIM sampler
    ddim_sampler = DDIMSampler(pipe.unet, (beta_start, beta_end), T)

    # Move the pipeline components to GPU
    pipe.unet.to("cuda")
    pipe.text_encoder.to("cuda")
    pipe.vae.to("cuda")
    pipe.scheduler = ddim_sampler.to("cuda")

    # Move tokenizer to GPU (if needed)
    pipe.tokenizer.to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    generator = torch.manual_seed(42)  # For reproducibility
    image = pipe(prompt, generator=generator).images[0]

    image.save(f"/home/sharifm/students/benshapira/advanced-dl--assignment-2/stable-diffusion-2-1/astronaut_rides_horse_T_{T}.png")
