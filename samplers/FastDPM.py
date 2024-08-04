import os
import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

def _map_gpu(gpu):
    if gpu == 'cuda':
        return lambda x: x.cuda()
    else:
        return lambda x: x.to(torch.device('cuda:' + gpu))

map_gpu = _map_gpu('cuda')


class FastDPM:
    def __init__(self, model, T=200, beta_0=0.0001, beta_T=0.02, steps=50, schedule='linear', approx_diff='STEP', kappa=1.0, batchsize=5, img_size=32, **kwargs):
        # Set default values for attributes
        self.model = model
        self.T = T
        self.steps = steps
        self.scheduler_type = schedule
        self.approx_diff = approx_diff
        self.kappa = kappa
        self.batchsize = batchsize
        self.img_size = img_size
        self.diffusion_config = {
            "beta_0": beta_0,
            "beta_T": beta_T,
            "T": T,
        }

    def rescale(self, X, batch=True):
        if not batch:
            return (X - X.min()) / (X.max() - X.min())
        else:
            for i in range(X.shape[0]):
                X[i] = self.rescale(X[i], batch=False)
            return X

    def calc_diffusion_hyperparams(self, T, beta_0, beta_T):
        Beta = torch.linspace(beta_0, beta_T, T)
        Alpha = 1 - Beta
        Alpha_bar = Alpha + 0
        Beta_tilde = Beta + 0
        for t in range(1, T):
            Alpha_bar[t] *= Alpha_bar[t - 1]
            Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])
        Sigma = torch.sqrt(Beta_tilde)

        _dh = {}
        _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
        return _dh

    def bisearch(self, f, domain, target, eps=1e-8):
        sign = -1 if target < 0 else 1
        left, right = domain
        for _ in range(1000):
            x = (left + right) / 2
            if f(x) < target:
                right = x
            elif f(x) > (1 + sign * eps) * target:
                left = x
            else:
                break
        return x

    def get_VAR_noise(self, S, schedule='linear'):

        target = np.prod(1 - np.linspace(self.diffusion_config["beta_0"], self.diffusion_config["beta_T"], self.diffusion_config["T"]))

        if schedule == 'linear':
            g = lambda x: np.linspace(self.diffusion_config["beta_0"], x, S)
            domain = (self.diffusion_config["beta_0"], 0.99)
        elif schedule == 'quadratic':
            g = lambda x: np.array([self.diffusion_config["beta_0"] * (1 + i * x) ** 2 for i in range(S)])
            domain = (0.0, 0.95 / np.sqrt(self.diffusion_config["beta_0"]) / S)
        else:
            raise NotImplementedError

        f = lambda x: np.prod(1 - g(x))
        largest_var = self.bisearch(f, domain, target, eps=1e-4)
        return g(largest_var)

    def get_STEP_step(self, S, schedule='linear'):
        if schedule == 'linear':
            c = (self.diffusion_config["T"] - 1.0) / (S - 1.0)
            list_tau = [np.floor(i * c) for i in range(S)]
        elif schedule == 'quadratic':
            list_tau = np.linspace(0, np.sqrt(self.diffusion_config["T"] * 0.8), S) ** 2
        else:
            raise NotImplementedError

        return [int(s) for s in list_tau]

    def std_normal(self, size):
        return map_gpu(torch.normal(0, 1, size=size))

    def _log_gamma(self, x):
        y = x - 1
        return np.log(2 * np.pi * y) / 2 + y * (np.log(y) - 1) + np.log(1 + 1 / (12 * y))

    def _log_cont_noise(self, t, beta_0, beta_T, T):
        delta_beta = (beta_T - beta_0) / (T - 1)
        _c = (1.0 - beta_0) / delta_beta
        t_1 = t + 1
        return t_1 * np.log(delta_beta) + self._log_gamma(_c + 1) - self._log_gamma(_c - t_1 + 1)

    # Standard DDPM generation
    def STD_sampling(self, model, X_t, size, diffusion_hyperparams):
        _dh = diffusion_hyperparams
        T, Alpha, Alpha_bar, Beta = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Beta"]
        assert len(Alpha_bar) == T
        assert len(size) == 4

        Sigma = _dh["Sigma"]

        with torch.no_grad():
            for t in range(T - 1, -1, -1):
                diffusion_steps = t * map_gpu(torch.ones(size[0]))
                epsilon_theta = model(X_t, diffusion_steps)
                X_t = (X_t - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
                if t > 0:
                    X_t = X_t + Sigma[t] * self.std_normal(size)
        return X_t

    # STEP
    def STEP_sampling(self, model, X_t, size, diffusion_hyperparams, user_defined_steps, kappa):
        _dh = diffusion_hyperparams
        T, Alpha, Alpha_bar, _ = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
        assert len(Alpha_bar) == T
        assert len(size) == 4
        assert 0.0 <= kappa <= 1.0

        T_user = len(user_defined_steps)
        user_defined_steps = sorted(list(user_defined_steps), reverse=True)

        with torch.no_grad():
            for i, tau in enumerate(user_defined_steps):
                diffusion_steps = tau * map_gpu(torch.ones(size[0]))
                epsilon_theta = model(X_t, diffusion_steps)
                if i == T_user - 1:  # the next step is to generate x_0
                    assert tau == 0
                    alpha_next = torch.tensor(1.0)
                    sigma = torch.tensor(0.0)
                else:
                    alpha_next = Alpha_bar[user_defined_steps[i + 1]]
                    sigma = kappa * torch.sqrt(
                        (1 - alpha_next) / (1 - Alpha_bar[tau]) * (1 - Alpha_bar[tau] / alpha_next))
                X_t *= torch.sqrt(alpha_next / Alpha_bar[tau])
                c = torch.sqrt(1 - alpha_next - sigma ** 2) - torch.sqrt(1 - Alpha_bar[tau]) * torch.sqrt(
                    alpha_next / Alpha_bar[tau])
                X_t += c * epsilon_theta + sigma * self.std_normal(size)
        return X_t

    # VAR
    def _precompute_VAR_steps(self, diffusion_hyperparams, user_defined_eta):
        _dh = diffusion_hyperparams
        T, Alpha, Alpha_bar, Beta = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Beta"]
        assert len(Alpha_bar) == T

        # compute diffusion hyperparameters for user defined noise
        T_user = len(user_defined_eta)
        Beta_tilde = map_gpu(torch.from_numpy(user_defined_eta)).to(torch.float32)
        Gamma_bar = 1 - Beta_tilde
        for t in range(1, T_user):
            Gamma_bar[t] *= Gamma_bar[t - 1]

        assert Gamma_bar[0] <= Alpha_bar[0] and Gamma_bar[-1] >= Alpha_bar[-1]

        continuous_steps = []
        with torch.no_grad():
            for t in range(T_user - 1, -1, -1):
                t_adapted = None
                for i in range(T - 1):
                    if Alpha_bar[i] >= Gamma_bar[t] > Alpha_bar[i + 1]:
                        t_adapted = self.bisearch(
                            f=lambda _t: self._log_cont_noise(_t, Beta[0].cpu().numpy(), Beta[-1].cpu().numpy(), T),
                            domain=(i - 0.01, i + 1.01),
                            target=np.log(Gamma_bar[t].cpu().numpy()))
                        break
                if t_adapted is None:
                    t_adapted = T - 1
                continuous_steps.append(t_adapted)  # must be decreasing
        return continuous_steps

    def VAR_sampling(self, model, X_t, size, diffusion_hyperparams, user_defined_eta, kappa, continuous_steps):
        _dh = diffusion_hyperparams
        T, Alpha, Alpha_bar, Beta = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Beta"]
        assert len(Alpha_bar) == T
        assert len(size) == 4
        assert 0.0 <= kappa <= 1.0

        # compute diffusion hyperparameters for user defined noise
        T_user = len(user_defined_eta)
        Beta_tilde = map_gpu(torch.from_numpy(user_defined_eta)).to(torch.float32)
        Gamma_bar = 1 - Beta_tilde
        for t in range(1, T_user):
            Gamma_bar[t] *= Gamma_bar[t - 1]

        assert Gamma_bar[0] <= Alpha_bar[0] and Gamma_bar[-1] >= Alpha_bar[-1]

        # print('begin sampling, total number of reverse steps = %s' % T_user)

        with torch.no_grad():
            for i, tau in enumerate(continuous_steps):
                diffusion_steps = tau * map_gpu(torch.ones(size[0]))
                epsilon_theta = model(X_t, diffusion_steps)
                if i == T_user - 1:  # the next step is to generate x_0
                    assert abs(tau) < 0.1
                    alpha_next = torch.tensor(1.0)
                    sigma = torch.tensor(0.0)
                else:
                    alpha_next = Gamma_bar[T_user - 1 - i - 1]
                    sigma = kappa * torch.sqrt(
                        (1 - alpha_next) / (1 - Gamma_bar[T_user - 1 - i]) * (
                                    1 - Gamma_bar[T_user - 1 - i] / alpha_next))
                X_t *= torch.sqrt(alpha_next / Gamma_bar[T_user - 1 - i])
                c = torch.sqrt(1 - alpha_next - sigma ** 2) - torch.sqrt(1 - Gamma_bar[T_user - 1 - i]) * torch.sqrt(
                    alpha_next / Gamma_bar[T_user - 1 - i])
                X_t += c * epsilon_theta + sigma * self.std_normal(size)

        return X_t

    def sample(self, X_t):
        # map diffusion hyperparameters to gpu
        diffusion_hyperparams = self.calc_diffusion_hyperparams(**self.diffusion_config)
        for key in diffusion_hyperparams:
            if key is not "T":
                diffusion_hyperparams[key] = map_gpu(diffusion_hyperparams[key])

        # sample
        if self.approx_diff == 'STD':
            x = self.STD_sampling(self.model, X_t, (self.batchsize, 1, self.img_size, self.img_size), diffusion_hyperparams)
        elif self.approx_diff == 'STEP':
            user_defined_steps = self.get_STEP_step(self.steps, self.scheduler_type)
            x = self.STEP_sampling(self.model, X_t,(self.batchsize, 1, self.img_size, self.img_size),
                               diffusion_hyperparams,
                               user_defined_steps,
                               kappa=self.kappa)
        elif self.approx_diff == 'VAR':
            user_defined_eta = self.get_VAR_noise(self.steps, self.scheduler_type)
            continuous_steps = self._precompute_VAR_steps(diffusion_hyperparams, user_defined_eta)
            x = self.VAR_sampling(self.model, X_t, (self.batchsize, 1, self.img_size, self.img_size),
                              diffusion_hyperparams,
                              user_defined_eta,
                              kappa=self.kappa,
                              continuous_steps=continuous_steps)

        return self.rescale(x)
