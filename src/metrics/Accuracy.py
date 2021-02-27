# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/metrics/Accuracy.py


import numpy as np
import math
from scipy import linalg
from tqdm import tqdm

from utils.sample import sample_latents
from utils.losses import latent_optimise

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

def batch_correct(outputs, labels):
    return (outputs.argmax(1) == labels).float().sum()

def calculate_accuracy(dataloader, generator, discriminator, D_loss, num_evaluate, truncated_factor, prior, latent_op,
                       latent_op_step, latent_op_alpha, latent_op_beta, device, cr, logger, eval_generated_sample=False):
    data_iter = iter(dataloader)
    batch_size = dataloader.batch_size
    disable_tqdm = device != 0
    discriminator.eval()
    generator.eval()

    if isinstance(generator, DataParallel) or isinstance(generator, DistributedDataParallel):
        z_dim = generator.module.z_dim
        num_classes = generator.module.num_classes
        conditional_strategy = discriminator.module.conditional_strategy
    else:
        z_dim = generator.z_dim
        num_classes = generator.num_classes
        conditional_strategy = discriminator.conditional_strategy

    total_batch = num_evaluate // batch_size
    print('Total Batch:', total_batch)

    if eval_generated_sample:
        logger.info("Calculating accuracies for real and fake images....")
        cum_fake_acc = 0.0
        cum_real_acc = 0.0
        cum_total = 0.0
        for batch_id in tqdm(range(total_batch), disable = False):
            zs, fake_labels = sample_latents(prior, batch_size, z_dim, truncated_factor, num_classes, None, device)
            if latent_op:
                zs = latent_optimise(zs, fake_labels, generator, discriminator, conditional_strategy, latent_op_step,
                                     1.0, latent_op_alpha, latent_op_beta, False, device)

            real_images, real_labels = next(data_iter)
            real_images, real_labels = real_images.to(device), real_labels.to(device)

            fake_images = generator(zs, fake_labels, evaluation=True)

            with torch.no_grad():
                dis_out_fake = discriminator(fake_images, fake_labels)
                dis_out_real = discriminator(real_images, real_labels)

            cum_real_acc += batch_correct(dis_out_real, real_labels)
            cum_fake_acc += batch_correct(dis_out_fake, fake_labels)
            cum_total += real_images.size(0)
        
        print('Test Acc:', cum_real_acc.item() / cum_total)
        print('Fake Acc:', cum_fake_acc.item() / cum_total)

        discriminator.train()
        generator.train()

        return cum_real_acc.item() / cum_total, cum_fake_acc.item() / cum_total
    else:
        logger.info("Calculating accuracies for real images....")
        cum_real_acc = 0.0
        cum_total = 0.0
        for batch_id in tqdm(range(total_batch), disable = False):
            real_images, real_labels = next(data_iter)
            real_images, real_labels = real_images.to(device), real_labels.to(device)

            with torch.no_grad():
                dis_out_real = discriminator(real_images, real_labels)
            
            cum_real_acc += batch_correct(dis_out_real, real_labels)
            cum_total += real_images.size(0)
        
        print('Test Acc:', cum_real_acc.item() / cum_total)
        discriminator.train()
        generator.train()

        return cum_real_acc.item() / cum_total
        