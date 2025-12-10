import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
from loguru import logger
import os
import wandb
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from torch.cuda.amp import GradScaler, autocast
from model import Ma_models
from models.OCT_encoder import OCT_Encoder
from omegaconf import OmegaConf
from loader import PatientDataGenerator #,get_sampler
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def find_model(model_name):
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    return checkpoint


def find_model_model(model_name):
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    checkpoint = checkpoint["models"]
    return checkpoint


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    logger.add(f"{logging_dir}/log.txt", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
    return logger

def cosine_similarity_loss(original_features, generated_features):
    """计算两个特征向量之间的余弦相似性损失"""
    original_features = original_features / original_features.norm(dim=-1, keepdim=True)
    generated_features = generated_features / generated_features.norm(dim=-1, keepdim=True)
    loss = 1 - F.cosine_similarity(original_features, generated_features).mean()
    return loss

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    scaler = GradScaler()
    torch.manual_seed(args.global_seed)
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)

    if args.wandb:
        wandb.init(project=args.model.replace('/', '_'))
        wandb.config = {
            "learning_rate": 0.0001,
            "epochs": args.epochs,
            "batch_size": args.global_batch_size,
            "dt-rank": args.dt_rank,
            "d-state": args.d_state,
            "save-path": experiment_dir,
            "autocast": args.autocast,
        }
    logger.info(f"Experiment directory created at {experiment_dir}")

    # 创建模型
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = Ma_models[args.model](
        input_size=latent_size,
        dt_rank=args.dt_rank,
        d_state=args.d_state,
    ).to(device)

    if args.init_from_pretrain_ckpt:
        model_state_dict_ = find_model_model(args.pretrain_ckpt_path)
        model.load_state_dict(model_state_dict_)
        ema = deepcopy(model).to(device)
        ema_state_dict_ = find_model(args.pretrain_ckpt_path)
        ema.load_state_dict(ema_state_dict_)
        logger.info(f"Loaded pretrain models from {args.pretrain_ckpt_path}")
    else:
        ema = deepcopy(model).to(device)

    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained('pretrain').to(device)

    # 加载 OCT 编码器
    oct_encoder = OCT_Encoder(
        img_size=args.image_size // 8,
        patch_size=int(args.model[-1]),
        in_channels=4,
        embed_dim=512,
        contain_mask_token=True,
    ).to(device)
    oct_ckpt_path = args.oct_ckpt
    oct_state_dict = find_model(oct_ckpt_path)
    oct_encoder.load_state_dict(oct_state_dict)
    oct_encoder.eval()

    lr = args.lr_ if args.init_from_pretrain_ckpt else args.lr
    prompt_octa = nn.Parameter(torch.randn(1,512).repeat(args.global_batch_size, 1).to(device), requires_grad=True)
    parameters = list(model.parameters())
    parameters.append(prompt_octa)
    opt = torch.optim.AdamW(parameters, lr=lr, weight_decay=0)

    # 创建数据加载器
    train_dataset = PatientDataGenerator(
        r'/mnt/h/OCT/OCTAP-600/FULL',
        r'/mnt/h/OCT/OCTAP-600/OCT',
        transform=transforms.Compose([
            transforms.CenterCrop([400, 400]),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.global_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True
    )

    logger.info(f"Dataset contains {len(train_dataset)}.")

    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()

    train_steps = args.init_train_steps if args.init_from_pretrain_ckpt else 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    logger.info(f"Training for {args.epochs} epochs...")
    epoch_list = []
    loss_list_CFP = []

    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        # for oct, octa in train_loader:
        #     oct = oct[:, :, 0, :, :]
        #     oct = torch.cat([oct] * 3, dim=1).to(device)
        #     for i in range(oct.shape[2]):
        #         oct_2d = torch.cat([oct[:, :, i, :, :]] * 3, dim=1).to(device)
        #         with torch.no_grad():
        #             if not torch.all((oct_2d >= -1) & (oct_2d <= 1)):
        #                 oct_2d = ((oct_2d - oct_2d.min()) * 1.0 / (oct_2d.max() - oct_2d.min())) * 2.0 - 1.0
        #             oct_2d = vae.encode(oct_2d).latent_dist.sample().mul_(0.18215)
        #             x_ = vae.encode(oct).latent_dist.sample().mul_(0.18215)
        #             x_ct_2 = oct_encoder(x_)
        #         t = torch.randint(0, diffusion.num_timesteps, (oct_2d.shape[0],), device=device)
        #         model_kwargs = dict(y=prompt_oct[:, i, :], y2=x_ct_2)
        item = 0
        for x_oct, octa in train_loader:
            item += 1
            x_oct = torch.cat([x_oct] * 3, dim=1)
            octa = torch.cat([octa] * 3, dim=1)
            x_oct = x_oct.to(device)
            octa = octa.to(device)
            with torch.no_grad():
                if not torch.all((octa >= -1) & (octa <= 1)):
                    octa = ((octa - octa.min()) * 1.0 / (octa.max() - octa.min())) * 2.0 - 1.0
                octa = vae.encode(octa).latent_dist.sample().mul_(0.18215)
                x_ = vae.encode(x_oct).latent_dist.sample().mul_(0.18215)
                x_oct_2 = oct_encoder(x_)
            t = torch.randint(0, diffusion.num_timesteps, (octa.shape[0],), device=device)
            model_kwargs = dict(y=prompt_octa, y2=x_oct_2)
            with autocast(enabled=args.autocast):
                loss_dict = diffusion.training_losses(model, octa, t, model_kwargs)  # 主要是 MSE 损失值，表示模型在每个时间步上预测的噪声和实际噪声之间的均方误差。
                loss = loss_dict["loss"].mean()
            if args.wandb:
                wandb.log({"loss": loss.item()})

            if torch.isnan(loss).any():
                logger.info(f"nan...... ignore losses......")
                continue

            with autocast(enabled=args.autocast):
                scaler.scale(loss).backward()

            if train_steps % args.accumulation_steps == 0:
                scaler.step(opt)
                scaler.update()
                update_ema(ema, model)
                opt.zero_grad()

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            epoch_list.append(epoch + 1)
            loss_list_CFP.append(running_loss)

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                logger.info(
                    f"(Epoch={epoch:05d}, step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "models": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                    "prompt_octa": prompt_octa
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()
    logger.info("Done!")
    if args.wandb:
        wandb.finish()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Enable WandB.", default=False)
    parser.add_argument("--autocast", action="store_true", help="Whether to use half-precision training.",
                        default=False)
    parser.add_argument('--config', type=str, help='Path to the configuration file',
                        default="config/CPGMD.yaml")
    args = parser.parse_args()

    cli_config = OmegaConf.create({k: v for k, v in args.__dict__.items() if v is not None and k != 'config'})
    args = OmegaConf.merge(OmegaConf.load(args.config), cli_config)
    main(args)
