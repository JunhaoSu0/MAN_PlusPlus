"""
ResNet backbone with MAN++ local learning (K=4).

Split points (K=4):
  Block 0: conv1 + bn1 + relu + maxpool + layer1
  Block 1: layer2
  Block 2: layer3
  Block 3: layer4  (no auxiliary head — direct to detection head)

Auxiliary detection heads on blocks 0, 1, 2 use EMA + LB + learnable
scale s. EMA source for block j is the first residual unit of block j+1.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ResNetMANPP",
    "resnet50_manpp", "resnet101_manpp", "resnet152_manpp",
    # Backward-compatible legacy MAN names.
    "ResNetMAN",
    "resnet50_man", "resnet101_man", "resnet152_man",
]

model_urls = {
    "resnet50":  "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


# ───────────────────────── building blocks ──────────────────────────
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride, 1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1   = norm_layer(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1   = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2   = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3   = norm_layer(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ──────────────────── ResNet-MAN++ backbone ────────────────────────
class ResNetMANPP(nn.Module):
    """
    ResNet with MAN++ local learning, K=4.
    During training the forward is called once per local block (stateful).
    During inference the full backbone runs in a single call and returns
    the FPN feature dict {C3, C4, C5}.
    """

    def __init__(self, block, layers, num_classes=80, momentum=0.995,
                 groups=1, width_per_group=64, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes    = 64
        self.groups      = groups
        self.base_width  = width_per_group
        self.momentum    = momentum
        self.num_classes = num_classes

        # ---- stem ----
        self.conv1   = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1     = norm_layer(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # ---- 4 stages ----
        self.layer1 = self._make_layer(block,  64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # channel counts after each stage
        self.C2_channels = 64  * block.expansion   # after layer1
        self.C3_channels = 128 * block.expansion   # after layer2
        self.C4_channels = 256 * block.expansion   # after layer3
        self.C5_channels = 512 * block.expansion   # after layer4

        # ---- MAN++ components for blocks 0, 1, 2 ----
        # EMA source: first residual unit of the *next* stage
        self.LB      = nn.ModuleList()
        self.EMA_Net = nn.ModuleList()
        self.ema_s   = nn.ParameterList()

        ema_sources = [self.layer2[0], self.layer3[0], self.layer4[0]]

        for i in range(3):
            lb  = copy.deepcopy(ema_sources[i])
            ema = copy.deepcopy(ema_sources[i])
            for p in ema.parameters():
                p.requires_grad = False
            self.LB.append(lb)
            self.EMA_Net.append(ema)
            self.ema_s.append(nn.Parameter(torch.tensor(1.0)))

        # ---- stateful counter for block-wise training ----
        self.block_idx = 0

        self._init_weights()

    # ----------------------------------------------------------------
    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample,
                        self.groups, self.base_width, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ----------------------------------------------------------------
    @torch.no_grad()
    def _ema_update(self, idx):
        """EMA: backbone next-stage first unit -> aux EMA_Net[idx]."""
        src = [self.layer2[0], self.layer3[0], self.layer4[0]][idx]
        for p_ema, p_src in zip(self.EMA_Net[idx].parameters(), src.parameters()):
            p_ema.data.mul_(self.momentum).add_(p_src.data, alpha=1 - self.momentum)

    # ----------------------------------------------------------------
    def forward_train(self, x, targets, aux_loss_fn, local_aux_backward=True,
                      aux_backward_fn=None):
        """
        Called repeatedly for each local block.
        Returns:
          - For blocks 0-2: (detached_features, aux_loss)
          - For block 3:    dict of FPN features {"C3": ..., "C4": ..., "C5": ...}
        """
        stages = [self.layer1, self.layer2, self.layer3, self.layer4]
        idx = self.block_idx

        # stem only on first block
        if idx == 0:
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = stages[idx](x)

        if idx < 3:
            # MAN++ auxiliary head: y = s * LB(x) + (2-s) * EMA(x)
            s = self.ema_s[idx]
            y = s * self.LB[idx](x) + (2 - s) * self.EMA_Net[idx](x)
            aux_loss = aux_loss_fn(idx, y, targets)
            if local_aux_backward:
                if aux_backward_fn is None:
                    aux_loss.backward()
                else:
                    aux_backward_fn(aux_loss)

            self._ema_update(idx)
            self.block_idx += 1
            return x.detach(), aux_loss.detach()

        else:
            # last block — return nothing; caller collects features
            self.block_idx = 0
            return x, None

    # ----------------------------------------------------------------
    def forward_inference(self, x):
        """Full forward, returns FPN-ready feature dict."""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"C3": c3, "C4": c4, "C5": c5}

    def forward_train_full(self, images, targets, aux_loss_fn,
                           aux_backward_fn=None):
        """
        Run all 4 blocks sequentially (as in local learning).
        Returns (feature_dict, total_aux_loss).
        """
        x = images
        total_aux = torch.tensor(0.0, device=images.device)
        for blk in range(4):
            if blk < 3:
                x, aux_loss = self.forward_train(
                    x, targets, aux_loss_fn,
                    aux_backward_fn=aux_backward_fn)
                total_aux = total_aux + aux_loss
            else:
                x, _ = self.forward_train(
                    x, targets, aux_loss_fn,
                    aux_backward_fn=aux_backward_fn)

        # Now x is C5. We need C3, C4 too for FPN.
        # Re-derive them from stored block outputs.
        # Actually we need to collect them during the pass.
        # Let's refactor: use a dedicated method.
        raise NotImplementedError("Use forward_train_collect instead.")

    def forward_train_collect(self, images, targets, aux_loss_fn,
                              local_aux_backward=True, aux_backward_fn=None):
        """
        Run all 4 local blocks. Backward aux losses along the way.
        Returns: feature_dict {"C3", "C4", "C5"}, total_aux_loss (for logging).
        """
        total_aux = torch.tensor(0.0, device=images.device)
        feats = {}

        x = self.maxpool(self.relu(self.bn1(self.conv1(images))))

        # Block 0: layer1
        x = self.layer1(x)
        s0 = self.ema_s[0]
        y0 = s0 * self.LB[0](x) + (2 - s0) * self.EMA_Net[0](x)
        aux0 = aux_loss_fn(0, y0, targets)
        if local_aux_backward:
            if aux_backward_fn is None:
                aux0.backward()
            else:
                aux_backward_fn(aux0)
        self._ema_update(0)
        total_aux = total_aux + aux0.detach()
        x = x.detach()

        # Block 1: layer2
        x = self.layer2(x)
        feats["C3"] = x
        s1 = self.ema_s[1]
        y1 = s1 * self.LB[1](x) + (2 - s1) * self.EMA_Net[1](x)
        aux1 = aux_loss_fn(1, y1, targets)
        if local_aux_backward:
            if aux_backward_fn is None:
                aux1.backward()
            else:
                aux_backward_fn(aux1)
        self._ema_update(1)
        total_aux = total_aux + aux1.detach()
        x = x.detach()
        feats["C3"] = feats["C3"].detach()

        # Block 2: layer3
        x = self.layer3(x)
        feats["C4"] = x
        s2 = self.ema_s[2]
        y2 = s2 * self.LB[2](x) + (2 - s2) * self.EMA_Net[2](x)
        aux2 = aux_loss_fn(2, y2, targets)
        if local_aux_backward:
            if aux_backward_fn is None:
                aux2.backward()
            else:
                aux_backward_fn(aux2)
        self._ema_update(2)
        total_aux = total_aux + aux2.detach()
        x = x.detach()
        feats["C4"] = feats["C4"].detach()

        # Block 3: layer4 (no aux head)
        x = self.layer4(x)
        feats["C5"] = x

        return feats, total_aux

    def forward(self, x, target=None, local_aux_backward=None, aux_loss_fn=None,
                aux_backward_fn=None):
        if local_aux_backward is None:
            local_aux_backward = self.training and target is not None and torch.is_grad_enabled()
        local_aux_backward = bool(local_aux_backward and torch.is_grad_enabled())

        if target is not None:
            if local_aux_backward:
                if aux_loss_fn is None:
                    raise ValueError("aux_loss_fn is required for MAN++ COCO local detection loss")
                return self.forward_train_collect(
                    x, target, aux_loss_fn, local_aux_backward=True,
                    aux_backward_fn=aux_backward_fn)
            return self.forward_inference(x), x.new_zeros(())

        return self.forward_inference(x)

    def load_pretrained(self, arch):
        """Load ImageNet pre-trained weights (skip fc)."""
        import torch.utils.model_zoo as model_zoo
        state = model_zoo.load_url(model_urls[arch])
        own = self.state_dict()
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
        self.load_state_dict(own, strict=False)


# ──────────────────────── factory functions ─────────────────────────
def resnet50_manpp(pretrained=False, **kw):
    m = ResNetMANPP(Bottleneck, [3, 4, 6, 3], **kw)
    if pretrained:
        m.load_pretrained("resnet50")
    return m

def resnet101_manpp(pretrained=False, **kw):
    m = ResNetMANPP(Bottleneck, [3, 4, 23, 3], **kw)
    if pretrained:
        m.load_pretrained("resnet101")
    return m

def resnet152_manpp(pretrained=False, **kw):
    m = ResNetMANPP(Bottleneck, [3, 8, 36, 3], **kw)
    if pretrained:
        m.load_pretrained("resnet152")
    return m


# Backward-compatible legacy MAN aliases.
ResNetMAN = ResNetMANPP

def resnet50_man(pretrained=False, **kw):
    return resnet50_manpp(pretrained=pretrained, **kw)

def resnet101_man(pretrained=False, **kw):
    return resnet101_manpp(pretrained=pretrained, **kw)

def resnet152_man(pretrained=False, **kw):
    return resnet152_manpp(pretrained=pretrained, **kw)
