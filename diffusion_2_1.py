import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler, DDPMScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
for step in [5, 10, 50, 100]:
    for sampler_type in ['DDIM', 'DDPM', 'DPMSolver++', 'fastDPM']:
        if sampler_type == 'DPMSolver++':
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif sampler_type == 'DDIM':
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif sampler_type == 'DDPM':
            pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        else:
            pipe.scheduler = None

        pipe = pipe.to("cuda")

        prompt = "a photo of an Israely swimmer dunking a basketball in the olympics"
        image = pipe(prompt, num_inference_steps=step).images[0]

        image.save(f"/home/sharifm/students/benshapira/advanced-dl--assignment-2/stable-diffusion-2-1/swimmer_{sampler_type}_steps{step}.png")
