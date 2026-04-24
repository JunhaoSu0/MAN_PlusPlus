"""
RetinaNet + MAN++ (K=4) training on COCO with DDP.

Usage:
  # Single node, 8 GPUs
  torchrun --nproc_per_node=8 train_coco.py /data/coco \
      --backbone resnet50 --epochs 100 --batch-size 8 --lr 4e-4

  # With ImageNet pretrained backbone
  torchrun --nproc_per_node=8 train_coco.py /data/coco \
      --backbone resnet101 --pretrained --epochs 100

Supported backbones: resnet50, resnet101, resnet152
"""

import os, argparse, random, shutil, time, math, warnings
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from networks.resnet_manpp import (
    resnet50_manpp, resnet101_manpp, resnet152_manpp,
)
from networks.retinanet import RetinaNet

# ────────────────────────── CLI ──────────────────────────────────────
parser = argparse.ArgumentParser("RetinaNet-MAN++ COCO")
parser.add_argument("data", metavar="DIR", help="COCO root (contains train2017/, val2017/, annotations/)")
parser.add_argument("--backbone", default="resnet50",
                    choices=["resnet50", "resnet101", "resnet152"])
parser.add_argument("--pretrained", action="store_true", help="use ImageNet pretrained backbone")
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("-b", "--batch-size", default=8, type=int, help="batch size per GPU")
parser.add_argument("--lr", default=4e-4, type=float)
parser.add_argument("--weight-decay", default=1e-4, type=float)
parser.add_argument("--momentum-manpp", "--momentum-man", dest="momentum_manpp",
                    default=0.995, type=float, help="EMA momentum for MAN++")
parser.add_argument("--num-classes", default=80, type=int)
parser.add_argument("--img-size", default=640, type=int)
parser.add_argument("-j", "--workers", default=8, type=int)
parser.add_argument("--print-freq", default=50, type=int)
parser.add_argument("--output", default="./outputs", type=str)
parser.add_argument("--resume", default="", type=str)
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--dist-backend", default="nccl")
parser.add_argument("--dist-url", default="env://")
args = parser.parse_args()

best_loss = float("inf")


# ────────────────────────── COCO Dataset ─────────────────────────────
class CocoDetection(torch.utils.data.Dataset):
    """Minimal COCO dataset wrapper. Requires pycocotools."""

    def __init__(self, root, ann_file, transforms=None, img_size=640):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(ann_file)
        self.ids  = sorted(self.coco.getImgIds())
        self.transforms = transforms
        self.img_size = img_size

        # build contiguous id mapping
        cats = sorted(self.coco.getCatIds())
        self.cat2label = {c: i for i, c in enumerate(cats)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        from PIL import Image
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.root, info["file_name"])
        img = Image.open(path).convert("RGB")
        ow, oh = img.size

        # resize
        img = img.resize((self.img_size, self.img_size))
        if self.transforms is not None:
            img = self.transforms(img)

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w < 1 or h < 1:
                continue
            # rescale to img_size
            x1 = x / ow * self.img_size
            y1 = y / oh * self.img_size
            x2 = (x + w) / ow * self.img_size
            y2 = (y + h) / oh * self.img_size
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat2label[a["category_id"]])

        target = {
            "boxes":  torch.as_tensor(boxes,  dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.long),
        }
        return img, target


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, 0), list(targets)


# ────────────────────────── main ─────────────────────────────────────
def main():
    global best_loss
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    args.gpu        = int(os.environ.get("LOCAL_RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = args.world_size > 1
    main_worker(args.gpu)


def main_worker(gpu):
    global best_loss
    if args.distributed:
        dist.init_process_group(args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=gpu)
        print(f"[Rank {gpu}] DDP init OK (world={args.world_size})", flush=True)

    torch.cuda.set_device(gpu)

    # ---- model ----
    backbone_fn = {
        "resnet50":  resnet50_manpp,
        "resnet101": resnet101_manpp,
        "resnet152": resnet152_manpp,
    }[args.backbone]

    backbone = backbone_fn(
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        momentum=args.momentum_manpp,
    )
    model = RetinaNet(backbone, num_classes=args.num_classes).cuda(gpu)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], find_unused_parameters=True)

    # ---- optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- resume ----
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=f"cuda:{gpu}")
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_loss   = ckpt.get("best_loss", float("inf"))
        print(f"=> resume from epoch {start_epoch}")

    cudnn.benchmark = True

    # ---- data ----
    tfm = T.Compose([T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])])

    train_set = CocoDetection(
        os.path.join(args.data, "train2017"),
        os.path.join(args.data, "annotations", "instances_train2017.json"),
        transforms=tfm, img_size=args.img_size)
    val_set = CocoDetection(
        os.path.join(args.data, "val2017"),
        os.path.join(args.data, "annotations", "instances_val2017.json"),
        transforms=tfm, img_size=args.img_size)

    train_sampler = DistributedSampler(train_set) if args.distributed else None
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    # ---- training loop ----
    loss_history = []
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_lr(optimizer, epoch)

        train_loss = train_one_epoch(train_loader, model, optimizer, epoch, gpu)
        val_loss = validate(val_loader, model, gpu)

        if gpu == 0:
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            loss_history.append(val_loss)
            os.makedirs(args.output, exist_ok=True)
            np.savetxt(os.path.join(args.output, "val_loss.txt"),
                       np.array(loss_history))
            save_ckpt({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
            }, is_best, args.output)
            print(f"Epoch {epoch} | val_loss={val_loss:.4f} | best={best_loss:.4f}")


# ────────────────────────── train / val ──────────────────────────────
def train_one_epoch(loader, model, optimizer, epoch, gpu):
    model.train()
    meter = AverageMeter()
    end = time.time()
    for i, (images, targets) in enumerate(loader):
        images = images.cuda(gpu, non_blocking=True)
        targets = [{k: v.cuda(gpu) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        out = model(images, targets)
        loss = out["cls_loss"] + out["reg_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        meter.update(loss.item(), images.size(0))
        if i % args.print_freq == 0 and gpu == 0:
            print(f"[Ep{epoch}][{i}/{len(loader)}] "
                  f"cls={out['cls_loss'].item():.4f} "
                  f"reg={out['reg_loss'].item():.4f} "
                  f"aux={out['aux_loss']:.4f} "
                  f"total={meter.avg:.4f} "
                  f"({time.time()-end:.1f}s)", flush=True)
            end = time.time()
    return meter.avg


@torch.no_grad()
def validate(loader, model, gpu):
    model.eval()
    meter = AverageMeter()
    for images, targets in loader:
        images = images.cuda(gpu, non_blocking=True)
        targets = [{k: v.cuda(gpu) for k, v in t.items()} for t in targets]
        out = model(images, targets, compute_loss=True, local_aux_backward=False)
        loss = out["cls_loss"] + out["reg_loss"]
        meter.update(loss.item(), images.size(0))
    return meter.avg


# ────────────────────────── utils ────────────────────────────────────
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count


def adjust_lr(optim, epoch):
    lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
    for g in optim.param_groups:
        g["lr"] = lr


def save_ckpt(state, is_best, out_dir):
    path = os.path.join(out_dir, "checkpoint.pth.tar")
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, os.path.join(out_dir, "model_best.pth.tar"))


if __name__ == "__main__":
    main()
