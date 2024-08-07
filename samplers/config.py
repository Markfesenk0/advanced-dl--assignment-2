# Configuration for different samplers
sampler_config = {
    "DDPM": {
        "beta_1": 1e-4,
        "beta_T": 0.025,
        "T": 200,
    },
    "DDIM": {
        "beta_1": 0.00009,
        "beta_T": 0.04,
        "T": 200,
    },
    "DPM_pp": {
        "beta_1": 0.0005,
        "beta_T": 0.05,
        "T": 200,
    },
    "FastDPM": {
        "beta_1": 0.0002,
        "beta_T": 0.04,
        "T": 200,
    }
}

def get_sampler_config(sampler_name):
    config = sampler_config.get(sampler_name, None)

    if config is None:
        raise ValueError(f"Sampler {sampler_name} not found in config")

    config["mean_type"] = "epsilon"
    config["var_type"] = "fixedlarge"
    config["steps"] = 200

    return config