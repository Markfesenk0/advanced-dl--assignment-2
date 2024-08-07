# report number of parameters
from torch import nn
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL

def print_num_params(model: nn.Module):
    """
    Return the number of parameters in the model.
    """
    n_params = sum(p.numel() for p in model.parameters())

    print("[[ ", "number of parameters: %.2fM" % (n_params / 1e6,), " ]]")
    return n_params


def get_text_embeds(prompt, device, dtype=torch.float16):
    model_id = "stabilityai/stable-diffusion-2-1-base"

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)

    # Tokenize text and get embeddings
    text_input = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors='pt')
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0].to(dtype)

    return text_embeddings


def latents_to_img(latents, device):
    # batch of latents -> list of images
    SCALING_FACTOR = 0.18215
    model_id = "stabilityai/stable-diffusion-2-1-base"

    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)

    latents = (1 / SCALING_FACTOR) * latents
    with torch.no_grad(): image = vae.decode(latents).sample
    # image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    images = [image for image in images]
    return images
