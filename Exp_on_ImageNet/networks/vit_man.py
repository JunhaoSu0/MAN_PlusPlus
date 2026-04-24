# -*- coding: utf-8 -*-
# Vision Transformer + MAN++ / MANPP K=4
from functools import partial
import copy
import torch
import torch.nn as nn


MANPP_K = 4


# -------- Drop-Path ---------------------------------------------------------
def drop_path(x, p: float = 0., training: bool = False):
    if p == 0. or not training:
        return x
    keep = 1 - p
    mask = keep + torch.rand(
        (x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device
    )
    mask.floor_()
    return x.div(keep) * mask


class DropPath(nn.Module):
    def __init__(self, p=0.): super().__init__(); self.p = p
    def forward(self, x):     return drop_path(x, self.p, self.training)


# -------- Patch Embedding ---------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img=224, patch=16, in_c=3, dim=768, norm_layer=None):
        super().__init__()
        self.img = img
        self.proj = nn.Conv2d(in_c, dim, patch, patch)
        self.norm = norm_layer(dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        assert H == W == self.img, 'unexpected image size'
        x = self.proj(x).flatten(2).transpose(1, 2)   # B N C
        return self.norm(x)


# -------- Attention & MLP ---------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.h = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.qkv   = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.adrop = nn.Dropout(attn_drop)
        self.proj  = nn.Linear(dim, dim)
        self.pdrop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, C // self.h)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # 3 * B h N d
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.adrop(attn.softmax(-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.pdrop(self.proj(x))


class MLP(nn.Module):
    def __init__(self, dim, ratio=4., drop=0.):
        super().__init__()
        hid = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., dp_ratio=0.):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads, qkv_bias, attn_drop, drop)
        self.dp  = DropPath(dp_ratio) if dp_ratio else nn.Identity()
        self.n2   = nn.LayerNorm(dim, eps=1e-6)
        self.mlp  = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.dp(self.attn(self.n1(x)))
        x = x + self.dp(self.mlp(self.n2(x)))
        return x


class MANPPAuxClassifier(nn.Module):
    """MAN++ auxiliary head: s * LB(x) + (2 - s) * EMA(x), then LN + FC."""
    def __init__(self, lb_block: nn.Module, ema_block: nn.Module,
                 classes: int, dim: int):
        super().__init__()
        self.lb_block  = lb_block          # learnable bias branch
        self.ema_block = ema_block         # EMA branch (frozen)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, classes)
        self.s    = nn.Parameter(torch.tensor(1.0))  # learnable scale per group

    def forward(self, features):
        y = self.s * self.lb_block(features) + (2 - self.s) * self.ema_block(features)
        return self.head(self.norm(y)[:, 0])


class MANPPVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3,
                 num_classes=1000, embed_dim=768,
                 depth=12, num_heads=12, groups=4,
                 mlp_ratio=4., drop_path_ratio=0.,
                 momentum=0.995):
        super().__init__()
        if groups != MANPP_K:
            raise ValueError(
                f'ViT MAN++ open-source implementation supports K={MANPP_K} '
                f'only; got groups={groups}.'
            )
        if depth % groups != 0:
            raise ValueError(f'depth ({depth}) must be divisible by K/groups ({groups}).')
        self.grp  = groups
        self.bp_g = depth // groups
        self.gidx = 0
        self.momentum = momentum
        self.manpp_k = groups
        self.num_aux_stages = groups - 1

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch = PatchEmbed(img_size, patch_size, in_c, embed_dim, norm_layer)
        n_patch = (img_size // patch_size) ** 2
        self.cls  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos  = nn.Parameter(torch.zeros(1, n_patch + 1, embed_dim))
        self.pdrop= nn.Dropout(0.)

        dpr = torch.linspace(0, drop_path_ratio, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, True, 0., 0., dpr[i])
            for i in range(depth)
        ])

        # For each of the first (groups-1) groups, build LB + EMA branches.
        # The EMA source is the first Block of the *next* group.
        self.aux_lb = nn.ModuleList()   # learnable copy
        self.aux_ema = nn.ModuleList()  # EMA copy (frozen)
        self.aux_clf = nn.ModuleList()

        for g in range(groups - 1):
            next_first = (g + 1) * self.bp_g   # index of first block in next group
            lb_block  = copy.deepcopy(self.blocks[next_first])
            ema_block = copy.deepcopy(self.blocks[next_first])
            # freeze EMA branch
            for p in ema_block.parameters():
                p.requires_grad = False
            self.aux_lb.append(lb_block)
            self.aux_ema.append(ema_block)
            self.aux_clf.append(
                MANPPAuxClassifier(lb_block, ema_block, num_classes, embed_dim)
            )

        self.norm  = norm_layer(embed_dim)
        self.head  = nn.Linear(embed_dim, num_classes)
        self.crit  = nn.CrossEntropyLoss()

        nn.init.trunc_normal_(self.pos, std=.02)
        nn.init.trunc_normal_(self.cls, std=.02)
        self.apply(self._init_w)

    @staticmethod
    def _init_w(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.zeros_(m.bias)

    # ---- helpers ----
    def _embed(self, x):
        x = self.patch(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        return self.pdrop(torch.cat((cls, x), 1) + self.pos)

    @torch.no_grad()
    def _ema_update(self, g):
        """Update EMA block for group g from the backbone's next-group first block."""
        next_first = (g + 1) * self.bp_g
        src = self.blocks[next_first]
        for p_ema, p_src in zip(self.aux_ema[g].parameters(), src.parameters()):
            p_ema.data.mul_(self.momentum).add_(p_src.data, alpha=1 - self.momentum)

    # ---- forward ----
    def forward(self, x, target):
        target = target.to(x.device)

        # ---------------- Training / MAN++ ----------------
        if self.training:
            if self.gidx == 0:
                x = self._embed(x)

            # Run one group of backbone Blocks
            start = self.gidx * self.bp_g
            for i in range(self.bp_g):
                x = self.blocks[start + i](x)

            if self.gidx < self.grp - 1:
                # Aux loss with MAN++ head (EMA + LB + scale)
                aux_loss = self.crit(self.aux_clf[self.gidx](x), target)

                # EMA update: backbone next-group first block -> aux_ema[gidx]
                self._ema_update(self.gidx)

                self.gidx += 1
                return x.detach(), aux_loss, torch.tensor(self.gidx, device=x.device)

            # All groups done -> final classifier
            logits = self.head(self.norm(x[:, 0]))
            loss   = self.crit(logits, target)
            self.gidx = 0
            return logits, loss

        # ---------------- Inference ---------------------
        x = self._embed(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, 0]))


# -------- Factory -----------------------------------------------------------
def _build_manpp_vit(embed_dim, num_heads, num_classes=1000, **kw):
    kw.setdefault('num_classes', num_classes)
    kw.setdefault('embed_dim', embed_dim)
    kw.setdefault('num_heads', num_heads)
    kw.setdefault('groups', MANPP_K)
    return MANPPVisionTransformer(**kw)


def manpp_vit_tiny_patch16_224(num_classes=1000, **kw):
    return _build_manpp_vit(192, 3, num_classes=num_classes, **kw)


def manpp_vit_small_patch16_224(num_classes=1000, **kw):
    return _build_manpp_vit(384, 6, num_classes=num_classes, **kw)


def manpp_vit_base_patch16_224(num_classes=1000, **kw):
    return _build_manpp_vit(768, 12, num_classes=num_classes, **kw)


MANPP_VIT_FACTORIES = {
    'vit_tiny': manpp_vit_tiny_patch16_224,
    'vit_small': manpp_vit_small_patch16_224,
    'vit_base': manpp_vit_base_patch16_224,
}

# Backward-compatible aliases for older training commands and checkpoints.
AuxClassifier = MANPPAuxClassifier
VisionTransformer = MANPPVisionTransformer
vit_tiny_patch16_224 = manpp_vit_tiny_patch16_224
vit_small_patch16_224 = manpp_vit_small_patch16_224
vit_base_patch16_224 = manpp_vit_base_patch16_224

__all__ = [
    'MANPP_K',
    'MANPPAuxClassifier',
    'MANPPVisionTransformer',
    'MANPP_VIT_FACTORIES',
    'manpp_vit_tiny_patch16_224',
    'manpp_vit_small_patch16_224',
    'manpp_vit_base_patch16_224',
    'AuxClassifier',
    'VisionTransformer',
    'vit_tiny_patch16_224',
    'vit_small_patch16_224',
    'vit_base_patch16_224',
]
