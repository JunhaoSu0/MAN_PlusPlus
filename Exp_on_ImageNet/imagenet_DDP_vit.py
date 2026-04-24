import os
# ---------- NCCL 环境 ----------
os.environ['NCCL_IB_DISABLE']    = '1'
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
os.environ['NCCL_DEBUG']         = 'INFO'

import argparse, random, shutil, time, warnings, math, numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets   as datasets
from tqdm import tqdm

from networks.vit_manpp import MANPP_K, MANPP_VIT_FACTORIES

# ------------------ CLI ------------------
parser = argparse.ArgumentParser('ViT-MAN++ DDP')
parser.add_argument('data', metavar='DIR', help='ImageNet root')
parser.add_argument('--net', default='vit_tiny',
                    choices=sorted(MANPP_VIT_FACTORIES.keys()),
                    help='MAN++ ViT backbone')
parser.add_argument('-j', '--workers', default=16, type=int)
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--batch-size', '-b', default=128, type=int)
parser.add_argument('--lr',  default=0.001, type=float)
parser.add_argument('--weight-decay', '--wd', default=0.05, type=float)
parser.add_argument('--print-freq', '-p', default=10, type=int)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--dist-backend', default='nccl')
parser.add_argument('--dist-url',     default='env://')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--train_url', default='./outputs/')
parser.add_argument('--local_module_num', '--k', dest='local_module_num',
                    type=int, default=MANPP_K,
                    help=f'MAN++ K/local module count; only K={MANPP_K} is supported')
args = parser.parse_args()
if args.local_module_num != MANPP_K:
    parser.error(
        f'ViT MAN++ open-source release supports K={MANPP_K} only; '
        f'got --local_module_num/--k={args.local_module_num}.'
    )

best_acc1 = 0
acc_list  = []

# ------------------ main ------------------
def main():
    if args.seed is not None:
        random.seed(args.seed); torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('Deterministic mode - may slow down')

    args.gpu         = int(os.environ.get('LOCAL_RANK', 0))
    args.world_size  = int(os.environ.get('WORLD_SIZE', 1))
    args.distributed = args.world_size > 1
    main_worker(args.gpu)


def main_worker(gpu):
    global best_acc1
    if args.distributed:
        dist.init_process_group(args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=gpu)
        print(f'[Rank {gpu}] DDP init OK (world={args.world_size})', flush=True)

    torch.cuda.set_device(gpu)

    # -------- model ----------
    create_model = MANPP_VIT_FACTORIES[args.net]
    if gpu == 0:
        print(f'=> creating {args.net} MAN++ K={args.local_module_num}', flush=True)
    model = create_model(num_classes=1000, groups=args.local_module_num).cuda(gpu)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], find_unused_parameters=True)

    # -------- optim ----------
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay
    )

    # -------- checkpoint ----------
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=f'cuda:{gpu}')
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_acc1   = ckpt.get('best_acc1', 0)
        start_epoch = ckpt.get('epoch', 0)
        print(f"=> resume '{args.resume}' (epoch {start_epoch})")
    else:
        start_epoch = 0

    cudnn.benchmark = True

    # -------- Data --------
    traindir = os.path.join(args.data, 'train')
    valdir   = os.path.join(args.data, 'val')
    norm = transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])

    train_set = datasets.ImageFolder(
        traindir,
        transforms.Compose([transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(), norm]))
    val_set = datasets.ImageFolder(
        valdir,
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                            transforms.ToTensor(), norm]))

    train_sampler = (torch.utils.data.distributed.DistributedSampler(
                        train_set, num_replicas=args.world_size, rank=gpu)
                     if args.distributed else None)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, gpu); return

    # ==================  train / epoch loop  ==================
    for epoch in range(start_epoch, args.epochs):
        if args.distributed: train_sampler.set_epoch(epoch)
        adjust_lr(optimizer, epoch, gpu)

        train(train_loader, model, criterion, optimizer, epoch, gpu, args)
        acc1 = validate(val_loader, model, criterion, gpu)
        acc1 = float(acc1.cpu())

        if gpu == 0:
            is_best   = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            acc_list.append(acc1)
            os.makedirs(args.train_url, exist_ok=True)
            np.savetxt(os.path.join(args.train_url, 'accuracy.txt'),
                       np.array(acc_list))
            save_ckpt({'epoch': epoch+1,
                       'state_dict': model.state_dict(),
                       'best_acc1': best_acc1,
                       'optimizer': optimizer.state_dict()},
                      is_best,
                      os.path.join(args.train_url, 'checkpoint.pth.tar'))


def train(loader, model, criterion, optimizer, epoch, gpu, args):
    meter_t = AverageMeter('Time', ':6.3f')
    meter_l = AverageMeter('Loss', ':.4e')
    prog = ProgressMeter(len(loader), [meter_t, meter_l],
                         prefix=f'[R{gpu}|Ep{epoch}] ')

    manpp_model = unwrap_model(model)
    aux_stage_count = manpp_model.num_aux_stages

    model.train(); end = time.time()
    for i, (images, target) in enumerate(tqdm(loader)):
        images = images.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        feat = images
        for _ in range(aux_stage_count):
            feat, aux_loss, _ = model(feat, target)
            aux_loss.backward()

        output, cls_loss = model(feat, target)
        cls_loss.backward()

        optimizer.step()

        # 统计
        meter_l.update(cls_loss.item(), images.size(0))
        meter_t.update(time.time()-end); end=time.time()
        if i % args.print_freq == 0: prog.display(i)

# ------------------ val ------------------
@torch.no_grad()
def validate(loader, model, crit, gpu):
    meter_t = AverageMeter('Time', ':6.3f')
    meter_l = AverageMeter('Loss', ':.4e')
    meter_a1= AverageMeter('Acc@1', ':6.2f'); meter_a5 = AverageMeter('Acc@5', ':6.2f')
    prog = ProgressMeter(len(loader), [meter_t, meter_l, meter_a1, meter_a5],
                         prefix=f'[R{gpu} Val] ')

    model.eval(); end = time.time()
    for i, (img, tgt) in enumerate(tqdm(loader, desc=f'Val R{gpu}', leave=False)):
        img = img.cuda(gpu, non_blocking=True)
        tgt = tgt.cuda(gpu, non_blocking=True)
        out = model(img, tgt)          # eval 模式仅返回 logits
        loss = crit(out, tgt)
        acc1, acc5 = accuracy(out, tgt, topk=(1, 5))

        meter_l.update(loss.item(), img.size(0))
        meter_a1.update(acc1[0], img.size(0)); meter_a5.update(acc5[0], img.size(0))
        meter_t.update(time.time()-end); end=time.time()
        if i % args.print_freq == 0: prog.display(i)

    if gpu == 0:
        print(f'* Val Acc@1 {meter_a1.avg:.3f}  Acc@5 {meter_a5.avg:.3f}')
    return meter_a1.avg


# ------------------ utils ------------------
class _nullcontext:          # 兼容 Py3.7
    def __enter__(self): return None
    def __exit__(self, *exc): return False


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name, self.fmt = name, fmt; self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        tpl = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return tpl.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, n, meters, prefix=''):
        self.meters, self.prefix = meters, prefix
        self.fmt = f'[{{:{len(str(n))}d}}/{n}]'
    def display(self, batch):
        entries = [self.prefix + self.fmt.format(batch)] + [str(m) for m in self.meters]
        print('\t'.join(entries), flush=True)


def adjust_lr(optim, epoch, gpu):
    lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
    for g in optim.param_groups: g['lr'] = lr
    if gpu == 0: print(f'===> Epoch {epoch} lr={lr:.6f}')


def save_ckpt(state, is_best, fname):
    torch.save(state, fname)
    if is_best:
        shutil.copyfile(fname,
                        os.path.join(os.path.dirname(fname), 'model_best.pth.tar'))


def accuracy(out, tgt, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk); bs = tgt.size(0)
        _, pred = out.topk(maxk, 1, True, True); pred = pred.t()
        correct = pred.eq(tgt.unsqueeze(0).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100. / bs))
        return res


# ------------------ run ------------------
if __name__ == '__main__':
    main() 
