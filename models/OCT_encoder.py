import torch
import torch.nn as nn
class OCT_Encoder(nn.Module):
    def __init__(
        self,
        img_size=28,
        patch_size=2,
        in_channels=4,
        embed_dim=1024,
        contain_mask_token=True, 
        reduction_ratio=14,
    ):
        super().__init__()

        self.vision_embedding = VisionEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            contain_mask_token=contain_mask_token, 
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((int((img_size/patch_size)**2),1))
        self.max_pool = nn.AdaptiveMaxPool2d((int((img_size/patch_size)**2),1))

        self.fc = nn.Sequential(
            nn.Linear(int((img_size/patch_size)**2), int((img_size/patch_size)**2 / reduction_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int((img_size/patch_size)**2 / reduction_ratio), int((img_size/patch_size)**2))
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.softm = nn.Softmax()

        self.sigmoid = nn.Sigmoid()
    def forward(self, x: torch.Tensor):
        x = self.vision_embedding(x)
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)
        x_spa = self.sigmoid(torch.mul(max_out, avg_out))#
        x_channel = avg_out + max_out
        x_channel = (self.sigmoid(self.fc((x_channel).squeeze(-1)))).unsqueeze(-1)
        out_channel = self.norm(x * x_channel)
        x = self.norm(x)
        out_spa  = torch.mul(x_spa, x)
        out = torch.mul(out_spa , out_channel)  #
        out = torch.add(out, x)
        output = self.norm(out)
        return  output

class VisionEmbedding(nn.Module):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        contain_mask_token=False,
        prepend_cls_token=False,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (
            img_size[0] // patch_size[0]
        )
        self.patch_shape = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        if contain_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.mask_token = None

        if prepend_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

    def num_position_embeddings(self):
        if self.cls_token is None:
            return self.num_patches
        else:
            return self.num_patches + 1

    def forward(self, x, masked_position=None, **kwargs):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model"
            f" ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x).flatten(2).transpose(1, 2)

        batch_size, seq_len, _ = x.size()

        if masked_position is not None:
            assert self.mask_token is not None
            mask_token = self.mask_token.expand(batch_size, seq_len, -1)
            w = masked_position.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(
                batch_size, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        return x