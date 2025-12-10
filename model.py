import numpy as np
import math
from timm.models.layers import  to_2tuple
import torch
from torch import nn
from models.mamba import Mamba as ssm
from timm.models.vision_transformer import Attention
class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=2, stride=2, in_chans=4, embed_dim=512, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class TimestepEmbed(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class FinalLayer(nn.Module):

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size * 2, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class CPGMD(nn.Module):

    def __init__(
        self,
        input_size=28,
        patch_size=2,
        strip_size = 2,
        in_channels=4,
        hidden_size=512,
        depth=16,
        learn_sigma=True,
        dt_rank=16,
        d_state=16,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.input_size = input_size
        self.x_embedder = PatchEmbed(input_size, patch_size, strip_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbed(hidden_size)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
                CMBM_MambaBlock(token_list=z(int(self.input_size/self.patch_size), i),
                               origina_list=z(int(self.input_size/self.patch_size), i),
                               D_dim=hidden_size,
                               E_dim=hidden_size*2,
                               dim_inner=hidden_size*2,
                               dt_rank=dt_rank,
                               d_state=d_state,)
                for i in range(depth)
            ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):

        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, y2):
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y2 = torch.mean(y2, dim=1)
        c1 = t+y
        c2 = t + y2
        c = torch.cat((c1, c2), dim=1)
        block_outputs = []
        for i in range(self.depth):
            if i == 0:  
                x = self.blocks[i](x, c)
            elif i > self.depth/2: 
                skip_connection = block_outputs[self.depth-i-1]
                x = self.blocks[i](block_outputs[-1] + skip_connection, c)
            else: 
                x = self.blocks[i](block_outputs[-1], c)
            block_outputs.append(x)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x
    def forward_with_cfg(self, x, t, y, y2, w, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, y2, w)

        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def z1(n):
    matrix = [[0] * n for _ in range(n)]
    num = 1
    for i in range(n):
        if i % 2 == 0:
            for j in range(n):
                matrix[i][j] = num
                num += 1
        else:
            for j in range(n-1, -1, -1):
                matrix[i][j] = num
                num += 1
    return matrix

def z2(n):
    matrix = [[0] * n for _ in range(n)]
    num = 1
    for j in range(n):
        if j % 2 == 0:
            for i in range(n):
                matrix[i][j] = num
                num += 1
        else:
            for i in range(n-1, -1, -1):
                matrix[i][j] = num
                num += 1
    return matrix

def z3(n):
    matrix = z1(n)
    matrix = [row[::-1] for row in matrix]
    return matrix

def z4(n):
    matrix = z2(n)
    matrix = [col[::-1] for col in matrix]
    return matrix

def z(n: int, i: int):
    rearrange_list = []

    if i % 4 == 1:
        matrix = z1(n)
    elif i % 4 == 2:
        matrix = z2(n)
    elif i % 4 == 3:
        matrix = z3(n)
    else:
        matrix = z4(n)

    for row in matrix:
        for num in row:
            rearrange_list.append(num - 1)

    return rearrange_list

    index_mapping = {index: i for i, index in enumerate(rearrange_list)}
    original_order_indexes = [index_mapping[i] for i in range(len(rearrange_list))]
    return rearrange_list, original_order_indexes

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class CMBM_MambaBlock(nn.Module):
    def __init__(
        self,
        D_dim: int,
        E_dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
        token_list: list,
        origina_list: list,
    ):
        super().__init__()
        self.D_dim = D_dim
        self.E_dim = E_dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.token_list = token_list
        self.origina_list = origina_list
        self.norm1 = nn.LayerNorm(D_dim)
        self.attn = Attention(D_dim, num_heads=8, qkv_bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D_dim * 2, 3 * D_dim, bias=True),
        )
        self.bow = ssm(
                d_model=D_dim,
                d_state=d_state,
                d_conv=4,
                expand=2,
                token_list=self.token_list,
                origina_list = self.origina_list,
                )
        self.fullnetwork = nn.Sequential(
            nn.LayerNorm(2*D_dim),
            nn.Linear(2*D_dim, D_dim, bias=True),
            nn.SiLU()
        )

        self.initialize_weights()

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
        b, s, d = x.shape
        x_out = self.norm1(x)
        x_out = modulate(x_out, shift_msa, scale_msa)
        c_out = x_out
        c_out = self.attn(c_out)
        x_out = self.bow(x_out)
        combined_out = torch.cat([x_out, c_out], dim=-1)
        x_output = self.fullnetwork(combined_out)
        x = x + gate_msa.unsqueeze(1) * x_output
        return x
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

def CPGMD_B(**kwargs):
    return CPGMD(depth=8, hidden_size=512, patch_size=2, strip_size=2, **kwargs)#

Ma_models = {'CPGMD-B/2' : CPGMD_B}
