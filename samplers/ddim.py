import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def extract_coefficients(v, t, shape):
    """
    Extract coefficients at specified timesteps and reshape for broadcasting.
    """
    gathered = torch.gather(v, dim=0, index=t)
    gathered = gathered.to(device=t.device, dtype=torch.float32)
    return gathered.view([t.shape[0]] + [1] * (len(shape) - 1))


class DDIMSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, steps, mean_type='epsilon', var_type='fixedlarge'):
        super().__init__()
        self.model = model
        self.T = T
        self.steps = steps

        # Generate T steps of beta
        betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float32)
        alphas = 1.0 - betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step, prev_time_step, eta):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        alpha_t = extract_coefficients(self.alphas_cumprod, t, x_t.shape)
        alpha_t_prev = extract_coefficients(self.alphas_cumprod, prev_t, x_t.shape)

        epsilon_theta_t = self.model(x_t, t)

        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x_t)
        x_t_minus_one = (
            torch.sqrt(alpha_t_prev / alpha_t) * x_t +
            (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt((alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
            sigma_t * epsilon_t
        )
        return x_t_minus_one

    @torch.no_grad()
    def forward(self, x_t, eta=0.2):
        """
        Sample from the model.
        """
        step_intervals = self.T // self.steps
        time_steps = np.arange(0, self.T, step_intervals)
        time_steps = time_steps + 1
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])
        if self.steps == self.T:
            self.steps -= 1

        with tqdm(reversed(range(self.steps)), colour="#6565b5", total=self.steps) as sampling_steps:
            for i in sampling_steps:
                x_t = self.sample_one_step(x_t, time_steps[i], time_steps_prev[i], eta)

        return x_t