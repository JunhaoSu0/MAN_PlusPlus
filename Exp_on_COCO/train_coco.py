"""
RetinaNet + MAN++ (K=4) training on COCO with DDP.

Usage:
  # Single node, 8 GPUs
  torchrun --nproc_per_node=8 train_coco.py /data/coco \
      --backbone resnet50 --training-mode manpp --batch-size 2 --lr 0.01

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
parser.add_argument("--training-mode", default="manpp", choices=["manpp", "bp"],
                    help="manpp runs K=4 local auxiliary backward; bp runs end-to-end backprop")
parser.add_argument("--pretrained", action="store_true", help="use ImageNet pretrained backbone")
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--max-iters", default=90000, type=int,
                    help="maximum training iterations; set <=0 to train for all epochs")
parser.add_argument("--lr-steps", default=[60000, 80000], nargs="+", type=int,
                    help="iteration milestones for 10x learning-rate decay")
parser.add_argument("-b", "--batch-size", default=2, type=int, help="batch size per GPU")
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight-decay", default=1e-4, type=float)
parser.add_argument("--momentum-manpp", "--momentum-man", dest="momentum_manpp",
                    default=0.995, type=float, help="EMA momentum for MAN++")
parser.add_argument("--num-classes", default=80, type=int)
parser.add_argument("--img-size", default=600, type=int,
                    help="shorter-side image scale")
parser.add_argument("--max-size", default=1000, type=int,
                    help="maximum longer-side image scale")
parser.add_argument("--flip-prob", default=0.5, type=float,
                    help="training horizontal flip probability")
parser.add_argument("-j", "--workers", default=8, type=int)
parser.add_argument("--print-freq", default=50, type=int)
parser.add_argument("--eval-freq", default=10, type=int,
                    help="run validation every N epochs; also validates on the final epoch")
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

    def __init__(self, root, ann_file, transforms=None, img_size=600,
                 max_size=1000, train=False, flip_prob=0.5):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(ann_file)
        self.ids  = sorted(self.coco.getImgIds())
        self.transforms = transforms
        self.img_size = img_size
        self.max_size = max_size
        self.train = train
        self.flip_prob = flip_prob

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

        scale = self.img_size / min(ow, oh)
        if round(max(ow, oh) * scale) > self.max_size:
            scale = self.max_size / max(ow, oh)
        nw, nh = int(round(ow * scale)), int(round(oh * scale))
        img = img.resize((nw, nh))

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w < 1 or h < 1:
                continue
            x1 = x * scale
            y1 = y * scale
            x2 = (x + w) * scale
            y2 = (y + h) * scale
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat2label[a["category_id"]])

        if self.train and random.random() < self.flip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for b in boxes:
                x1, x2 = b[0], b[2]
                b[0] = nw - x2
                b[2] = nw - x1

        if self.transforms is not None:
            img = self.transforms(img)

        target = {
            "boxes":  torch.as_tensor(boxes,  dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.long),
        }
        return img, target


def collate_fn(batch):
    imgs, targets = zip(*batch)
    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)
    padded = imgs[0].new_zeros((len(imgs), imgs[0].shape[0], max_h, max_w))
    for i, img in enumerate(imgs):
        _, h, w = img.shape
        padded[i, :, :h, :w] = img
    return padded, list(targets)


def resolve_coco_image_dir(root, split):
    direct = os.path.join(root, split)
    nested = os.path.join(root, "images", split)
    if os.path.isdir(direct):
        return direct
    if os.path.isdir(nested):
        return nested
    raise FileNotFoundError(
        f"Cannot find COCO image directory for {split}: expected "
        f"{direct} or {nested}"
    )


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
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay)

    # ---- resume ----
    start_epoch = 0
    global_iter = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=f"cuda:{gpu}")
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        global_iter = ckpt.get("iteration", 0)
        best_loss   = ckpt.get("best_loss", float("inf"))
        print(f"=> resume from epoch {start_epoch}")

    cudnn.benchmark = True

    # ---- data ----
    tfm = T.Compose([T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])])

    train_set = CocoDetection(
        resolve_coco_image_dir(args.data, "train2017"),
        os.path.join(args.data, "annotations", "instances_train2017.json"),
        transforms=tfm, img_size=args.img_size, max_size=args.max_size,
        train=True, flip_prob=args.flip_prob)
    val_set = CocoDetection(
        resolve_coco_image_dir(args.data, "val2017"),
        os.path.join(args.data, "annotations", "instances_val2017.json"),
        transforms=tfm, img_size=args.img_size, max_size=args.max_size,
        train=False, flip_prob=0.0)

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

        train_loss, global_iter = train_one_epoch(
            train_loader, model, optimizer, epoch, gpu, global_iter)
        reached_max_iters = args.max_iters > 0 and global_iter >= args.max_iters
        do_eval = ((epoch + 1) % args.eval_freq == 0) or \
                  (epoch + 1 == args.epochs) or reached_max_iters

        if gpu == 0:
            os.makedirs(args.output, exist_ok=True)
            save_ckpt({
                "epoch": epoch + 1,
                "iteration": global_iter,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
            }, False, args.output)

        if do_eval:
            val_loss = validate(val_loader, model, gpu)
            if gpu == 0:
                is_best = val_loss < best_loss
                best_loss = min(val_loss, best_loss)
                loss_history.append(val_loss)
                np.savetxt(os.path.join(args.output, "val_loss.txt"),
                           np.array(loss_history))
                save_ckpt({
                    "epoch": epoch + 1,
                    "iteration": global_iter,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                }, is_best, args.output)
                print(f"Epoch {epoch} | iter={global_iter} "
                      f"| train_loss={train_loss:.4f} "
                      f"| val_loss={val_loss:.4f} | best={best_loss:.4f}")
        elif gpu == 0:
            print(f"Epoch {epoch} | iter={global_iter} "
                  f"| train_loss={train_loss:.4f} "
                  f"| validation skipped (eval_freq={args.eval_freq})")

        if reached_max_iters:
            break


# ────────────────────────── train / val ──────────────────────────────
def train_one_epoch(loader, model, optimizer, epoch, gpu, global_iter):
    model.train()
    meter = AverageMeter()
    end = time.time()
    for i, (images, targets) in enumerate(loader):
        if args.max_iters > 0 and global_iter >= args.max_iters:
            break
        images = images.cuda(gpu, non_blocking=True)
        targets = [{k: v.cuda(gpu) for k, v in t.items()} for t in targets]

        adjust_lr(optimizer, global_iter)
        optimizer.zero_grad(set_to_none=True)
        out = model(
            images, targets,
            local_aux_backward=(args.training_mode == "manpp"))
        loss = out["cls_loss"] + out["reg_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        global_iter += 1

        meter.update(loss.item(), images.size(0))
        if i % args.print_freq == 0 and gpu == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"[Ep{epoch}][{i}/{len(loader)}][iter {global_iter}] "
                  f"cls={out['cls_loss'].item():.4f} "
                  f"reg={out['reg_loss'].item():.4f} "
                  f"aux={out['aux_loss']:.4f} "
                  f"lr={lr:.6f} "
                  f"total={meter.avg:.4f} "
                  f"({time.time()-end:.1f}s)", flush=True)
            end = time.time()
    return meter.avg, global_iter


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


def adjust_lr(optim, iteration):
    decay = sum(iteration >= step for step in args.lr_steps)
    lr = args.lr * (0.1 ** decay)
    for g in optim.param_groups:
        g["lr"] = lr


def save_ckpt(state, is_best, out_dir):
    path = os.path.join(out_dir, "checkpoint.pth.tar")
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, os.path.join(out_dir, "model_best.pth.tar"))


if __name__ == "__main__":
    main()
