import torch


class NoiseScheduleVP:
    def __init__(
            self,
            betas=None,
            alphas_cumprod=None,
            dtype=torch.float32,
    ):
        if betas is not None:
            log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
        else:
            assert alphas_cumprod is not None
            log_alphas = 0.5 * torch.log(alphas_cumprod)
        self.T = 1.
        self.log_alpha_array = self.numerical_clip_alpha(log_alphas).reshape((1, -1,)).to(dtype=dtype)
        self.total_N = self.log_alpha_array.shape[1]
        self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)

    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        log_sigmas = 0.5 * torch.log(1. - torch.exp(2. * log_alphas))
        lambs = log_alphas - log_sigmas
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t):
        return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device),
                              self.log_alpha_array.to(t.device)).reshape((-1))

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
        Delta = self.beta_0 ** 2 + tmp
        return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)


def model_wrapper(
        model,
        noise_schedule,
        steps=200.
):

    def get_model_input_time(t_continuous):
        # return (t_continuous - 1. / noise_schedule.total_N) * 1000.
        return (t_continuous / 1.) * steps

    def noise_pred_fn(x, t_continuous):
        t_input = get_model_input_time(t_continuous)
        return model(x, t_input)

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        return noise_pred_fn(x, t_continuous)

    return model_fn


class DPMSolverPP:
    def __init__(
            self,
            model_fn,
            noise_schedule
    ):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        return self.data_prediction_fn(x, t)

    def get_time_steps(self, t_T, t_0, N, device):
        return torch.linspace(t_T, t_0, N + 1).to(device)

    def denoise_to_zero_fn(self, x, s):
        return self.data_prediction_fn(x, s)

    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        ns = self.noise_schedule
        dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        phi_1 = torch.expm1(-h)
        if model_s is None:
            model_s = self.model_fn(x, s)
        x_t = (
                sigma_t / sigma_s * x
                - alpha_t * phi_1 * model_s
        )
        if return_intermediate:
            return x_t, {'model_s': model_s}
        else:
            return x_t

    def dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t):
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_1), ns.marginal_lambda(
            t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        phi_1 = torch.expm1(-h)
        x_t = (
                (sigma_t / sigma_prev_0) * x
                - (alpha_t * phi_1) * model_prev_0
                - 0.5 * (alpha_t * phi_1) * D1_0
        )
        return x_t

    def dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t):
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_2), ns.marginal_lambda(
            t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1. / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)
        phi_1 = torch.expm1(-h)
        phi_2 = phi_1 / h + 1.
        phi_3 = phi_2 / h - 0.5
        x_t = (
                (sigma_t / sigma_prev_0) * x
                - (alpha_t * phi_1) * model_prev_0
                + (alpha_t * phi_2) * D1
                - (alpha_t * phi_3) * D2
        )
        return x_t

    def dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order):
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.dpm_solver_second_update(x, model_prev_list, t_prev_list, t)
        elif order == 3:
            return self.dpm_solver_third_update(x, model_prev_list, t_prev_list, t)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, lower_order_final=True,
               denoise_to_zero=False, return_intermediate=False):

        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        device = x.device
        intermediates = []
        with torch.no_grad():
            assert steps >= order
            timesteps = self.get_time_steps(t_T=t_T, t_0=t_0, N=steps, device=device)
            assert timesteps.shape[0] - 1 == steps
            # Init the initial values.
            step = 0
            t = timesteps[step]
            t_prev_list = [t]
            model_prev_list = [self.model_fn(x, t)]
            if return_intermediate:
                intermediates.append(x)
            for step in range(1, order):
                t = timesteps[step]
                x = self.dpm_solver_update(x, model_prev_list, t_prev_list, t, step)
                if return_intermediate:
                    intermediates.append(x)
                t_prev_list.append(t)
                model_prev_list.append(self.model_fn(x, t))
            for step in range(order, steps + 1):
                t = timesteps[step]
                # We only use lower order for steps < 10
                if lower_order_final and steps < 10:
                    step_order = min(order, steps + 1 - step)
                else:
                    step_order = order
                x = self.dpm_solver_update(x, model_prev_list, t_prev_list, t, step_order)
                if return_intermediate:
                    intermediates.append(x)
                for i in range(order - 1):
                    t_prev_list[i] = t_prev_list[i + 1]
                    model_prev_list[i] = model_prev_list[i + 1]
                t_prev_list[-1] = t
                # We do not need to evaluate the final model value.
                if step < steps:
                    model_prev_list[-1] = self.model_fn(x, t)
            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x

def interpolate_fn(x, xp, yp):
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand