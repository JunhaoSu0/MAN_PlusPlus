"""
RetinaNet with FPN, focal loss, and box regression.
Works with the ResNet-MAN++ backbone that produces {"C3", "C4", "C5"}.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────── FPN ──────────────────────────────────────
class FPN(nn.Module):
    """Feature Pyramid Network producing P3–P7 from C3, C4, C5."""

    def __init__(self, C3_channels, C4_channels, C5_channels, out_channels=256):
        super().__init__()
        self.lateral5 = nn.Conv2d(C5_channels, out_channels, 1)
        self.lateral4 = nn.Conv2d(C4_channels, out_channels, 1)
        self.lateral3 = nn.Conv2d(C3_channels, out_channels, 1)

        self.smooth5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # P6 from C5, P7 from P6
        self.p6 = nn.Conv2d(C5_channels, out_channels, 3, stride=2, padding=1)
        self.p7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
        )

    def forward(self, feats):
        C3, C4, C5 = feats["C3"], feats["C4"], feats["C5"]

        P5 = self.lateral5(C5)
        P4 = self.lateral4(C4) + F.interpolate(P5, size=C4.shape[2:], mode="nearest")
        P3 = self.lateral3(C3) + F.interpolate(P4, size=C3.shape[2:], mode="nearest")

        P5 = self.smooth5(P5)
        P4 = self.smooth4(P4)
        P3 = self.smooth3(P3)

        P6 = self.p6(C5)
        P7 = self.p7(P6)

        return [P3, P4, P5, P6, P7]


# ────────────────────── Classification & Box Subnets ─────────────────
class ClassificationSubnet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        layers = []
        for _ in range(4):
            layers += [
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*layers)
        self.out  = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)

        # init
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.out.weight, 0, 0.01)
        nn.init.constant_(self.out.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x):
        return self.out(self.conv(x))


class RegressionSubnet(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        layers = []
        for _ in range(4):
            layers += [
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*layers)
        self.out  = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)

        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.out.weight, 0, 0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(self.conv(x))


# ────────────────────────── Anchors ──────────────────────────────────
class Anchors(nn.Module):
    """Generate anchors for all FPN levels."""

    def __init__(self, sizes=(32, 64, 128, 256, 512),
                 ratios=(0.5, 1.0, 2.0),
                 scales=(2 ** 0, 2 ** (1/3), 2 ** (2/3))):
        super().__init__()
        self.sizes  = sizes
        self.ratios = ratios
        self.scales = scales
        self.num_anchors = len(ratios) * len(scales)

    def _generate_base(self, size):
        anchors = []
        for r in self.ratios:
            for s in self.scales:
                h = size * s * math.sqrt(r)
                w = size * s / math.sqrt(r)
                anchors.append([-w/2, -h/2, w/2, h/2])
        return torch.tensor(anchors, dtype=torch.float32)

    def forward(self, features):
        """features: list of feature maps [P3..P7]."""
        all_anchors = []
        device = features[0].device
        for level, (feat, size) in enumerate(zip(features, self.sizes)):
            _, _, H, W = feat.shape
            base = self._generate_base(size).to(device)
            stride = 2 ** (3 + level)  # P3->8, P4->16, ...
            grid_x = torch.arange(W, device=device, dtype=torch.float32)
            grid_y = torch.arange(H, device=device, dtype=torch.float32)
            shift_x = (grid_x * stride + stride / 2).repeat(H)
            shift_y = (grid_y * stride + stride / 2).repeat_interleave(W)
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
            anchors = (shifts.unsqueeze(1) + base.unsqueeze(0)).reshape(-1, 4)
            all_anchors.append(anchors)
        return torch.cat(all_anchors, dim=0)


# ────────────────────────── Focal Loss ───────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, cls_preds, reg_preds, anchors, targets):
        """
        Args:
            cls_preds: (B, total_anchors, num_classes)
            reg_preds: (B, total_anchors, 4)
            anchors:   (total_anchors, 4)
            targets:   list[dict] with keys "boxes" (N,4) and "labels" (N,)
        Returns:
            cls_loss, reg_loss
        """
        batch_size = cls_preds.size(0)
        total_cls_loss = torch.tensor(0.0, device=cls_preds.device)
        total_reg_loss = torch.tensor(0.0, device=cls_preds.device)
        num_pos = 0

        for b in range(batch_size):
            gt_boxes  = targets[b]["boxes"]   # (M, 4)
            gt_labels = targets[b]["labels"]  # (M,)

            if gt_boxes.numel() == 0:
                total_cls_loss += torch.sigmoid(cls_preds[b]).sum() * 0
                continue

            ious = box_iou(anchors, gt_boxes)             # (A, M)
            max_iou, matched = ious.max(dim=1)            # (A,)

            # assign: pos >= 0.5, neg < 0.4, ignore between
            pos_mask    = max_iou >= 0.5
            neg_mask    = max_iou < 0.4
            ignore_mask = ~pos_mask & ~neg_mask

            n_pos = pos_mask.sum().item()
            num_pos += n_pos

            # classification focal loss
            cls_target = torch.zeros_like(cls_preds[b])
            if n_pos > 0:
                cls_target[pos_mask, gt_labels[matched[pos_mask]].long()] = 1

            p = torch.sigmoid(cls_preds[b])
            pt = p * cls_target + (1 - p) * (1 - cls_target)
            alpha_t = self.alpha * cls_target + (1 - self.alpha) * (1 - cls_target)
            fl = -alpha_t * (1 - pt) ** self.gamma * pt.clamp(min=1e-8).log()
            # zero out ignored
            if ignore_mask.any():
                fl[ignore_mask] = 0
            total_cls_loss += fl.sum()

            # regression smooth-L1 (only positive anchors)
            if n_pos > 0:
                pos_anchors = anchors[pos_mask]
                pos_gt      = gt_boxes[matched[pos_mask]]
                reg_target  = encode_boxes(pos_anchors, pos_gt)
                total_reg_loss += F.smooth_l1_loss(
                    reg_preds[b][pos_mask], reg_target, reduction="sum"
                )

        num_pos = max(num_pos, 1)
        return total_cls_loss / num_pos, total_reg_loss / num_pos


# ────────────────────── box utilities ────────────────────────────────
def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes (N,4) and (M,4)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-6)


def encode_boxes(anchors, gt_boxes):
    """Encode gt boxes relative to anchors (x1y1x2y2 format)."""
    a_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    a_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    a_w  = anchors[:, 2] - anchors[:, 0]
    a_h  = anchors[:, 3] - anchors[:, 1]

    g_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    g_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    g_w  = gt_boxes[:, 2] - gt_boxes[:, 0]
    g_h  = gt_boxes[:, 3] - gt_boxes[:, 1]

    dx = (g_cx - a_cx) / a_w.clamp(min=1)
    dy = (g_cy - a_cy) / a_h.clamp(min=1)
    dw = torch.log(g_w / a_w.clamp(min=1))
    dh = torch.log(g_h / a_h.clamp(min=1))
    return torch.stack([dx, dy, dw, dh], dim=1)


def decode_boxes(anchors, deltas):
    """Decode predicted offsets to x1y1x2y2 boxes."""
    a_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    a_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    a_w  = anchors[:, 2] - anchors[:, 0]
    a_h  = anchors[:, 3] - anchors[:, 1]

    cx = deltas[:, 0] * a_w + a_cx
    cy = deltas[:, 1] * a_h + a_cy
    w  = torch.exp(deltas[:, 2]) * a_w
    h  = torch.exp(deltas[:, 3]) * a_h

    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)


# ──────────────────── RetinaNet top-level ────────────────────────────
class RetinaNet(nn.Module):
    """
    Full RetinaNet = ResNet-MAN++ backbone + FPN + cls/reg subnets.
    During training, backbone runs local-learning (aux losses backpropped
    inside backbone.forward), then FPN + detection head loss is returned.
    """

    def __init__(self, backbone, num_classes=80, fpn_channels=256):
        super().__init__()
        self.backbone = backbone
        self.fpn = FPN(
            backbone.C3_channels,
            backbone.C4_channels,
            backbone.C5_channels,
            fpn_channels,
        )
        self.num_classes = num_classes
        self.anchors_gen = Anchors()
        self.num_anchors = self.anchors_gen.num_anchors
        self.cls_subnet = ClassificationSubnet(fpn_channels, self.num_anchors, num_classes)
        self.reg_subnet = RegressionSubnet(fpn_channels, self.num_anchors)
        self.focal_loss = FocalLoss()

    def forward(self, images, targets=None, compute_loss=None, local_aux_backward=None):
        """
        Args:
            images: (B, 3, H, W)
            targets: list[dict] each with "boxes" and "labels", or None
            compute_loss: if True, return detection losses even in eval mode
            local_aux_backward: if True, run MAN++ aux backward inside backbone
        Returns:
            losses: dict {"cls_loss", "reg_loss", "aux_loss"}
            preds:  dict {"cls_preds", "reg_preds", "anchors"}
        """
        B = images.size(0)
        if compute_loss is None:
            compute_loss = self.training and targets is not None
        if compute_loss and targets is None:
            raise ValueError("targets must be provided when compute_loss=True")

        if local_aux_backward is None:
            local_aux_backward = self.training and targets is not None and torch.is_grad_enabled()
        local_aux_backward = bool(local_aux_backward and torch.is_grad_enabled())

        if local_aux_backward:
            # Make pseudo-labels for aux heads (use majority class per image or 0)
            aux_labels = torch.zeros(B, dtype=torch.long, device=images.device)
            for i, t in enumerate(targets):
                if t["labels"].numel() > 0:
                    aux_labels[i] = t["labels"].mode().values.long()

            feats, aux_loss = self.backbone(
                images, aux_labels, local_aux_backward=True)
        else:
            feats = self.backbone(images)
            aux_loss = images.new_zeros(())

        fpn_feats = self.fpn(feats)
        anchors = self.anchors_gen(fpn_feats)

        cls_list, reg_list = [], []
        for f in fpn_feats:
            c = self.cls_subnet(f)
            r = self.reg_subnet(f)
            B_, _, H_, W_ = c.shape
            c = c.view(B_, self.num_anchors, self.num_classes, H_, W_)
            c = c.permute(0, 3, 4, 1, 2).reshape(B_, -1, self.num_classes)
            r = r.view(B_, self.num_anchors, 4, H_, W_)
            r = r.permute(0, 3, 4, 1, 2).reshape(B_, -1, 4)
            cls_list.append(c)
            reg_list.append(r)

        cls_preds = torch.cat(cls_list, dim=1)
        reg_preds = torch.cat(reg_list, dim=1)

        if compute_loss:
            cls_loss, reg_loss = self.focal_loss(cls_preds, reg_preds, anchors, targets)
            return {"cls_loss": cls_loss, "reg_loss": reg_loss, "aux_loss": aux_loss}

        return {"cls_preds": cls_preds, "reg_preds": reg_preds, "anchors": anchors}
