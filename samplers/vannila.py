import torch
from torch import nn
from torch.nn import functional as F


def extract_coefficients(v, t, shape):
    """
    Extract coefficients at specified timesteps and reshape for broadcasting.
    """
    gathered = torch.gather(v, dim=0, index=t).float()
    return gathered.view([t.shape[0]] + [1] * (len(shape) - 1))


class DDPMSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='epsilon', var_type='fixedlarge'):
        assert mean_type in ['xprev', 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod, [1, 0], value=1)[:-1]

        # Register buffers for diffusion process calculations
        self.register_buffer(
            'sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer(
            'sqrt_recip_minus_one_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        self.register_buffer(
            'posterior_variance',
            self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer(
            'posterior_log_variance_clipped',
            torch.log(torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_cumprod_prev) * self.betas / (1. - alphas_cumprod))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

    def calculate_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior.
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
                extract_coefficients(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
                extract_coefficients(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var = extract_coefficients(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var

    def predict_xstart_from_noise(self, x_t, t, noise):
        assert x_t.shape == noise.shape
        return (
                extract_coefficients(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_coefficients(self.sqrt_recip_minus_one_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_xstart_from_previous(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (
                extract_coefficients(1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
                extract_coefficients(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t
        )

    def compute_mean_variance(self, x_t, t):
        model_log_var = {
            'fixedlarge': torch.log(torch.cat([self.posterior_variance[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_variance_clipped,
        }[self.var_type]
        model_log_var = extract_coefficients(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t)
            x_0 = self.predict_xstart_from_previous(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t)
            model_mean, _ = self.calculate_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            noise = self.model(x_t, t)
            x_0 = self.predict_xstart_from_noise(x_t, t, noise=noise)
            model_mean, _ = self.calculate_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)

        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def forward(self, x_T):
        """
        Sample from the model (Algorithm 2 in the paper).
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.compute_mean_variance(x_t=x_t, t=t)
            noise = torch.randn_like(x_t) if time_step > 0 else 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        return torch.clip(x_t, -1, 1)
