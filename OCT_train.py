
import torch
import os
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from loader import PatientDataGenerator
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
import wandb
from glob import glob
from loguru import logger
from diffusers.models import AutoencoderKL
from time import time
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict
from models.OCT_encoder import OCT_Encoder
from omegaconf import OmegaConf



def create_logger(logging_dir):
    logger.add(f"{logging_dir}/log.txt", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
    return logger

def infoNCE_loss_b(input_tensor, tau=0.07):
    batch_size, seq_len, feat_dim = input_tensor.shape
    reshaped_tensor = input_tensor.reshape(batch_size, seq_len*feat_dim)
    reshaped_tensor = F.normalize(reshaped_tensor, p=2, dim=1)
    sim_matrix = torch.matmul(reshaped_tensor, reshaped_tensor.T) / tau
    labels = torch.arange(reshaped_tensor.size(0), dtype=torch.long, device=reshaped_tensor.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    scaler = GradScaler()
    seed = args.embedder_global_seed
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建实验目录
    if args.wandb:
        wandb.init(project="OCT_encoder")
        wandb.config = {"learning_rate": 0.0001, "epochs": args.embedder_epoch, "batch_size": args.embedder_global_batch_size}

    os.makedirs(args.embedder_results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.embedder_results_dir}/*"))
    model_string_name = "oct_encoder"
    experiment_dir = f"{args.embedder_results_dir}/{experiment_index:03d}-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    model = OCT_Encoder(img_size=28, patch_size=args.embedder_patch_size, in_channels=4, embed_dim=args.embedder_embed_dim, contain_mask_token=True).to(device)
    ema = deepcopy(model).to(device)
    vae = AutoencoderKL.from_pretrained("pretrain").to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    train_dataset = PatientDataGenerator(
        r'/mnt/h/OCT/OCTAP-600/FULL',
        r'/mnt/h/OCT/OCTAP-600/OCT',
        transform=transforms.Compose([
            transforms.CenterCrop([400,400]),
            transforms.Resize((224,224)),  # 调整图像大小
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])  # 转换为Tensor格式
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.global_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True
    )
    update_ema(ema, model, decay=0)
    ema.eval()
    model.train()

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    criterion_main = nn.CrossEntropyLoss()

    logger.info(f"Training for {args.embedder_epoch} epochs...")
    for epoch in range(args.embedder_epoch):
        logger.info(f"Beginning epoch {epoch}...")
        item=0

        for x_oct, _ in train_loader:
            item += 1
            x_oct = x_oct.to(device)
            x_oct = torch.cat([x_oct] * 3, dim=1)
            with torch.no_grad():
                x_oct = vae.encode(x_oct).latent_dist.sample().mul_(0.18215)
            opt.zero_grad()
            with autocast(enabled=args.autocast):
                x = model(x_oct)
            loss = infoNCE_loss_b(x)

            if args.wandb:
                wandb.log({"loss": loss.item()})

            if torch.isnan(loss).any():
                logger.info(f"nan...      ignore losses....")
                continue

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            update_ema(ema, model)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.8f}, Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.embedder_ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "models": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Enable WandB.", default=False)
    parser.add_argument("--autocast", action="store_true", help="Whether to use half-precision training.", default=False)
    parser.add_argument('--config', type=str, help='Path to the configuration file',
                        default='config/CPGMD.yaml')
    args = parser.parse_args()
    cli_config = OmegaConf.create({k: v for k, v in args.__dict__.items() if v is not None and k != 'config'})
    args = OmegaConf.merge(OmegaConf.load(args.config), cli_config)
    main(args)