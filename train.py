import argparse
import copy
import os
from functools import partial
import functools

import torch
import torchvision
from torchvision.transforms import transforms
from tqdm import trange

from models.unet import DDPMTrainObjective, UNet
from samplers.DPMSolverPP import NoiseScheduleVP, model_wrapper, DPMSolverPP
from samplers.FastDPM import FastDPM
from samplers.ddim import DDIMSampler
from samplers.vannila import DDPMSampler

import logging

logger = logging.getLogger(__name__)


# train utils:
def update_ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def repeat_samples(dataloader):
    """Repeat samples from dataloader indefinitely."""
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(warmup_param, step):
    return min(step, warmup_param) / warmup_param


# train logic:
def train(
        # (UNet) Architecture parameters:
        hid_channels_init: int = 128,  # base channel of UNet
        ch_mult: int = (1, 2, 2, 2),  # channel multiplier
        attn: int = (1,),  # add attention to these levels  # TODO what is it? original paper?
        num_res_blocks: int = 2,  # number of residual blocks (per level)
        dropout: float = 0.1,  # dropout rate of resblock

        # Gaussian Diffusion parameters:
        T: int = 200,  # number of time steps
        beta_1: float = 1e-4,  # start beta value
        beta_T: float = 0.02,  # end beta value
        mean_type: str = 'epsilon',  # predict variable
        var_type: str = 'fixedlarge',  # variance type

        # Training parameters:
        lr: float = 2e-4,  # target learning rate
        grad_clip: float = 1.0,  # gradient norm clipping
        total_steps: int = 30_000,  # total training steps  # TODO
        warmup_param: int = 5000,  # learning rate warmup  # TODO
        batch_size: int = 150,  # batch size
        ema_decay: float = 0.9999,  # ema decay rate
        save_every: int = 1000,  # frequency of saving checkpoints, 0 to disable during training
):
    device = torch.device('cuda:0')

    # dataset
    dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True, download=True,
        transform=transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Resize(32),
            transforms.Lambda(lambda x: (x - 0.5) * 2)
        ]))

    # Show first image for debug:
    # import matplotlib.pyplot as plt
    # plt.imshow(dataset[0][0].squeeze().numpy())
    # plt.show()

    input_shape = dataset[0][0].shape
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True)
    dataloader = repeat_samples(dataloader)

    # Model setup
    net_model = UNet(
        T=T, input_channels=input_shape[0], hid_channels_init=hid_channels_init,
        ch_mults=ch_mult, attn=attn,
        num_res_blocks=num_res_blocks, dropout=dropout)
    ema_model = copy.deepcopy(net_model).to(device)
    optim = torch.optim.Adam(net_model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(  # TODO what scheduler to use? (refer to official impl)
        optim,
        lr_lambda=partial(warmup_lr, warmup_param))
    model_objective = DDPMTrainObjective(net_model, beta_1, beta_T, T).to(device)

    # Training:
    for step in (pbar := trange(total_steps, desc="Training...")):
        optim.zero_grad()
        x_0 = next(dataloader).to(device)
        loss = model_objective(x_0).mean()
        loss.backward()

        # Update model:
        torch.nn.utils.clip_grad_norm_(  # TODO is it also used in the official impl.?
            net_model.parameters(), grad_clip)
        optim.step()
        sched.step()

        # Update the accumulating EMA model:
        update_ema(net_model, ema_model, ema_decay)

        # Log the loss
        # logger.info(f"Step {step}: loss={loss.item()}")
        pbar.set_postfix({'loss': loss.item()})

        if step % save_every == 0:
            # Save checkpoint
            ckpt = {
                'net_model': net_model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'sched': sched.state_dict(),
                'optim': optim.state_dict(),
                'step': step,
                'sampler_kwargs': dict(T=T, beta_1=beta_1, beta_T=beta_T,
                                       mean_type=mean_type, var_type=var_type),
            }
            torch.save(ckpt, os.path.join(logs_main_dir, 'ckpt.pt'))
            torch.save(ema_model,
                       os.path.join(logs_main_dir, 'ema_model.pt'))

            # Evaluate model
            evaluate(image_size=input_shape)


@torch.no_grad()
def evaluate(gen_batch_size=5, n_images=25, image_size=(1, 32, 32), sampler_type="DDPM", sampler_kwargs: dict = None, experiment_dir=None, steps=20):
    device = torch.device('cuda:0')

    # Load ema model
    ema_model = torch.load(os.path.join(logs_main_dir, 'ema_model.pt'))
    ema_model.eval()

    if sampler_kwargs is None:
        ckpt = torch.load(os.path.join(logs_main_dir, 'ckpt.pt'))
        sampler_kwargs = ckpt['sampler_kwargs']
    if sampler_type == "DDPM":
        sampler = DDPMSampler(ema_model, img_size=image_size[1], **sampler_kwargs).to(device)
    elif sampler_type == "DDIM":
        sampler = DDIMSampler(ema_model, **sampler_kwargs).to(device)
    elif sampler_type == "DPM_pp":
        # beta_1 = 0.0005, beta_T = 0.05
        sampler_kwargs['beta_1'] = 0.0005
        sampler_kwargs['beta_T'] = 0.05
        noise_schedule = NoiseScheduleVP(betas=torch.linspace(sampler_kwargs['beta_1'], sampler_kwargs['beta_T'], sampler_kwargs['T']).double())
        model_fn = model_wrapper(ema_model, noise_schedule=noise_schedule, T=sampler_kwargs['T'])
        dpm_sampler = DPMSolverPP(model_fn, noise_schedule)
        sampler = functools.partial(dpm_sampler.sample, steps=sampler_kwargs['steps'], denoise_to_zero=True, order=3)
    elif sampler_type == "FastDPM":
        sampler = FastDPM(ema_model, steps=sampler_kwargs['steps'], T=sampler_kwargs['T'], approx_diff='STEP', beta_0=sampler_kwargs['beta_1'], beta_T=sampler_kwargs['beta_T'], img_size=image_size[-1], batchsize=gen_batch_size)
        sampler = sampler.sample

    # Sample image with model
    images = []
    for i in range(n_images // gen_batch_size):
        print(f'working on batch {i} out of {n_images // gen_batch_size}')
        x_T = torch.randn((gen_batch_size, *image_size))
        batch_images = sampler(x_T.to(device)).detach().cpu()
        images.append(batch_images)
    images = torch.cat(images, dim=0)

    # [Quantitative]: TODO calculate scores
    # (IS, IS_std), FID = get_inception_and_fid_score(...)

    # [Qualitative]: Saved generated images
    # torchvision.utils.save_image(images,
    #                              os.path.join('/home/sharifm/students/benshapira/advanced-dl--assignment-2/logs/',
    #                                           f'gen_samples_{sampler_type}.png'), nrow=gen_batch_size)
    # save each image
    torchvision.utils.save_image(images,
                                 os.path.join('./logs/new', f'gen_samples_{sampler_type}_{steps}.png'),
                                 nrow=gen_batch_size)


if __name__ == '__main__':
    T = 200  # number of time steps
    steps = 10
    beta_1 = 0.0001  # start beta value
    beta_T = 0.02  # end beta value
    mean_type = 'epsilon'  # predict variable
    var_type = 'fixedlarge'  # variance type
    logs_main_dir = '/home/sharifm/students/markfesenko/projects/DLAT-HW2/logs/'

    sampler_kwargs = dict(T=T, beta_1=beta_1, beta_T=beta_T,
                          mean_type=mean_type, var_type=var_type, steps=steps)
    # train()
    for steps in [5, 10, 50, 200]:
        sampler_kwargs['steps'] = steps
        evaluate(sampler_type="FastDPM", sampler_kwargs=sampler_kwargs)
        evaluate(sampler_type="DPM++", sampler_kwargs=sampler_kwargs)
        evaluate(sampler_type="DDPM", sampler_kwargs=sampler_kwargs)
        evaluate(sampler_type="DDIM", sampler_kwargs=sampler_kwargs)

