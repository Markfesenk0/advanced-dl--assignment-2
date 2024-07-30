from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def extract(v, i, shape):
    """
    Get the i-th number in v, and the shape of v is mostly (T, ), the shape of i is mostly (batch_size, ).
    equal to [v[index] for index in i]
    """
    out = torch.gather(v, index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)

    # reshape to (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))
    return out


class DDIMSampler(nn.Module):
    def __init__(self, model, beta: Tuple[int, int], T: int, init_noise_sigma=1.0):
        super().__init__()
        self.model = model
        self.T = T

        beta_t = torch.linspace(*beta, T, dtype=torch.float32)
        alpha_t = 1.0 - beta_t
        self.register_buffer("alpha_t_bar", torch.cumprod(alpha_t, dim=0))

        # Add dtype attribute
        self.dtype = torch.float32

        # Add init_noise_sigma attribute
        self.init_noise_sigma = init_noise_sigma

    def set_timesteps(self, num_inference_steps, device=None):
        if device is None:
            device = self.alpha_t_bar.device

        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(
            0, self.T - 1, steps=num_inference_steps, device=device, dtype=torch.long
        ).to(device)

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step: int, prev_time_step: int, eta: float):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)

        # predict noise using model
        epsilon_theta_t = self.model(x_t, t)

        # calculate x_{t-1}
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x_t)
        x_t_minus_one = (
                torch.sqrt(alpha_t_prev / alpha_t) * x_t +
                (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                    (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
                sigma_t * epsilon_t
        )
        return x_t_minus_one

    def step(self, latents, t, **kwargs):
        prev_t = max(0, t - 1)
        return self.sample_one_step(latents, t, prev_t, eta=kwargs.get('eta', 0.0))

    @torch.no_grad()
    def forward(self, x_t, eta=0.0, only_return_x_0: bool = True, interval: int = 1):
        steps = self.num_inference_steps
        time_steps = self.timesteps
        time_steps_prev = torch.cat([torch.tensor([0]).to(time_steps.device), time_steps[:-1]])

        # if method == "linear":
        #     a = self.T // steps
        #     time_steps = np.asarray(list(range(0, self.T, a)))
        # elif method == "quadratic":
        #     time_steps = (np.linspace(0, np.sqrt(self.T * 0.8), steps) ** 2).astype(np.int)
        # else:
        #     raise NotImplementedError(f"sampling method {method} is not implemented!")
        #
        # # add one to get the final alpha values right (the ones from first scale to data during sampling)
        # time_steps = time_steps + 1
        # # previous sequence
        # time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        x = [x_t * self.init_noise_sigma]  # Use init_noise_sigma to scale initial noise
        with tqdm(reversed(range(steps)), colour="#6565b5", total=steps) as sampling_steps:
            for i in sampling_steps:
                x_t = self.sample_one_step(x_t, time_steps[i], time_steps_prev[i], eta)

                if not only_return_x_0 and ((steps - i) % interval == 0 or i == 0):
                    x.append(torch.clip(x_t, -1.0, 1.0))

                sampling_steps.set_postfix(ordered_dict={"step": i + 1, "sample": len(x)})

        if only_return_x_0:
            return x_t
        return torch.stack(x, dim=1)