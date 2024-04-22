# Adapted from https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/tree/master/code/train.py
import os
import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
from torch.utils.data.sampler import BatchSampler
import wandb
import ssl

import dataset
from dataset.Inshop import Inshop_Dataset
import loss
from net.resnet import *
from net.googlenet import *
from net.bn_inception import *
from utils import evaluate_cos, evaluate_cos_Inshop, evaluate_cos_SOP

ssl._create_default_https_context = ssl._create_unverified_context


parser = argparse.ArgumentParser(
    description="Official implementation of `Mean Field Theory in Deep Metric Learning` "
    + "Our code is modified from `https://github.com/dichotomies/proxy-nca` "
    + "and `https://github.com/tjddus9597/Proxy-Anchor-CVPR2020`."
)
# export directory, training and val datasets, test datasets
parser.add_argument("--LOG_DIR", default="../logs", help="Path to log folder")
parser.add_argument(
    "--dataset", default="cub", help="Training dataset, e.g. cub, cars, SOP, Inshop"
)
parser.add_argument(
    "--embedding-size",
    default=512,
    type=int,
    dest="sz_embedding",
    help="Size of embedding that is appended to backbone model.",
)
parser.add_argument(
    "--batch-size",
    default=150,
    type=int,
    dest="sz_batch",
    help="Number of samples per batch.",
)
parser.add_argument(
    "--epochs",
    default=60,
    type=int,
    dest="nb_epochs",
    help="Number of training epochs.",
)
parser.add_argument("--gpu-id", default=0, type=int, help="ID of GPU that is used for training.")
parser.add_argument(
    "--workers",
    default=4,
    type=int,
    dest="nb_workers",
    help="Number of workers for dataloader.",
)
parser.add_argument("--model", default="bn_inception", help="Model for training")
parser.add_argument("--loss", type=str, help="Criterion for training", required=True)
parser.add_argument("--optimizer", default="adamw", help="Optimizer setting")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate setting")
parser.add_argument("--lr-ratio", type=float, help="lr for proxy / lr for embedder")
parser.add_argument("--weight-decay", default=1e-4, type=float, help="Weight decay setting")
parser.add_argument("--lr-decay-step", default=10, type=int, help="Learning decay step setting")
parser.add_argument(
    "--lr-decay-gamma", default=0.5, type=float, help="Learning decay gamma setting"
)
parser.add_argument("--alpha", default=0.01, type=float, help="Scaling Parameter setting")
parser.add_argument("--beta", default=80, type=float, help="Scaling Parameter setting")
parser.add_argument("--mrg", default=0.8, type=float, help="Margin parameter setting")
parser.add_argument(
    "--pos-mrg", default=0.02, type=float, help="Margin parameter setting for positive pairs"
)
parser.add_argument(
    "--neg-mrg", default=0.3, type=float, help="Margin parameter setting for negative pairs"
)
parser.add_argument("--mf-reg", type=float, help="Mean field regularization parameter setting")
parser.add_argument("--IPC", type=int, help="Balanced sampling, images per class")
parser.add_argument("--warm", default=1, type=int, help="Warmup training epochs")
parser.add_argument(
    "--bn-freeze", default=1, type=int, help="Batch normalization parameter freeze"
)
parser.add_argument("--l2-norm", default=1, type=int, help="L2 normlization")
parser.add_argument("--remark", default="", help="Any reamrk")
parser.add_argument("--seed", default=1, type=int, help="Random seed")
parser.add_argument("--patience", default=-1, type=int, help="Patience for early stopping")
parser.add_argument(
    "--metric-to-monitor",
    default="mean_average_precision_at_r",
    help="Metric to monitor",
)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  # set random seed for all gpus


if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Directory for Log
LOG_DIR = (
    args.LOG_DIR
    + "/logs_{}/{}_{}_embedding{}_alpha{}_beta{}_mrg{}_pos_mrg{}_neg_mrg{}_{}_lr{}_lrr{}_batch{}_seed{}{}".format(
        args.dataset,
        args.model,
        args.loss,
        args.sz_embedding,
        args.alpha,
        args.beta,
        args.mrg,
        args.pos_mrg,
        args.neg_mrg,
        args.optimizer,
        args.lr,
        args.lr_ratio,
        args.sz_batch,
        args.seed,
        args.remark,
    )
)
# Wandb Initialization
wandb.init(project=f"MeanFieldTheoryInDeepMetricLearning_{args.dataset}", notes=LOG_DIR)
wandb.config.update(args)

os.chdir("../data/")
data_root = os.getcwd()
# Dataset Loader and Sampler
if args.dataset != "Inshop":
    trn_dataset = dataset.load(
        name=args.dataset,
        root=data_root,
        mode="train",
        transform=dataset.utils.make_transform(
            is_train=True, is_inception=(args.model == "bn_inception")
        ),
    )
else:
    trn_dataset = Inshop_Dataset(
        root=data_root,
        mode="train",
        transform=dataset.utils.make_transform(
            is_train=True, is_inception=(args.model == "bn_inception")
        ),
    )

if args.IPC:
    balanced_sampler = dataset.sampler.BalancedSampler(
        trn_dataset, batch_size=args.sz_batch, images_per_class=args.IPC
    )
    batch_sampler = BatchSampler(balanced_sampler, batch_size=args.sz_batch, drop_last=True)
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        num_workers=args.nb_workers,
        pin_memory=True,
        batch_sampler=batch_sampler,
    )
    print("Balanced Sampling")

else:
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        batch_size=args.sz_batch,
        shuffle=True,
        num_workers=args.nb_workers,
        drop_last=True,
        pin_memory=True,
    )
    print("Random Sampling")

if args.dataset != "Inshop":
    ev_dataset = dataset.load(
        name=args.dataset,
        root=data_root,
        mode="eval",
        transform=dataset.utils.make_transform(
            is_train=False, is_inception=(args.model == "bn_inception")
        ),
    )

    dl_ev = torch.utils.data.DataLoader(
        ev_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True,
    )

else:
    query_dataset = Inshop_Dataset(
        root=data_root,
        mode="query",
        transform=dataset.utils.make_transform(
            is_train=False, is_inception=(args.model == "bn_inception")
        ),
    )

    dl_query = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True,
    )

    gallery_dataset = Inshop_Dataset(
        root=data_root,
        mode="gallery",
        transform=dataset.utils.make_transform(
            is_train=False, is_inception=(args.model == "bn_inception")
        ),
    )

    dl_gallery = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True,
    )

nb_classes = trn_dataset.nb_classes()

# Backbone Model
if args.model.find("googlenet") + 1:
    model = googlenet(
        embedding_size=args.sz_embedding,
        pretrained=True,
        is_norm=args.l2_norm,
        bn_freeze=args.bn_freeze,
    )
elif args.model.find("bn_inception") + 1:
    model = bn_inception(
        embedding_size=args.sz_embedding,
        pretrained=True,
        is_norm=args.l2_norm,
        bn_freeze=args.bn_freeze,
    )
elif args.model.find("resnet18") + 1:
    model = Resnet18(
        embedding_size=args.sz_embedding,
        pretrained=True,
        is_norm=args.l2_norm,
        bn_freeze=args.bn_freeze,
    )
elif args.model.find("resnet50") + 1:
    model = Resnet50(
        embedding_size=args.sz_embedding,
        pretrained=True,
        is_norm=args.l2_norm,
        bn_freeze=args.bn_freeze,
    )
elif args.model.find("resnet101") + 1:
    model = Resnet101(
        embedding_size=args.sz_embedding,
        pretrained=True,
        is_norm=args.l2_norm,
        bn_freeze=args.bn_freeze,
    )
model = model.cuda()

if args.gpu_id == -1:
    model = nn.DataParallel(model)

# DML Losses
if args.loss == "ProxyAnchor":
    criterion = loss.ProxyAnchorLoss(nb_classes=nb_classes, sz_embed=args.sz_embedding).cuda()
elif args.loss == "ProxyNCALoss":
    criterion = loss.ProxyNCALoss(nb_classes=nb_classes, sz_embed=args.sz_embedding).cuda()
elif args.loss == "MultiSimilarity":
    criterion = loss.MultiSimilarityLoss().cuda()
elif args.loss == "Contrastive":
    criterion = loss.ContrastiveLoss(pos_margin=args.pos_mrg, neg_margin=args.neg_mrg).cuda()
elif args.loss == "Triplet":
    criterion = loss.TripletLoss().cuda()
elif args.loss == "NPair":
    criterion = loss.NPairLoss().cuda()
elif args.loss == "ClassWiseMultiSimilarity":
    criterion = loss.ClassWiseMultiSimilarityLoss(
        alpha=args.alpha,
        beta=args.beta,
        base=args.mrg,
    ).cuda()
elif args.loss == "MeanFieldContrastive":
    criterion = loss.MeanFieldContrastiveLoss(
        num_classes=nb_classes,
        embedding_size=args.sz_embedding,
        pos_margin=args.pos_mrg,
        neg_margin=args.neg_mrg,
        mf_reg=args.mf_reg,
        mf_power=2,
    ).cuda()
elif args.loss == "MeanFieldClassWiseMultiSimilarity":
    criterion = loss.MeanFieldClassWiseMultiSimilarityLoss(
        num_classes=nb_classes,
        embedding_size=args.sz_embedding,
        alpha=args.alpha,
        beta=args.beta,
        base=args.mrg,
        mf_reg=args.mf_reg,
        mf_power=2,
    ).cuda()


# Train Parameters
param_groups = [
    {
        "params": (
            list(set(model.parameters()).difference(set(model.model.embedding.parameters())))
            if args.gpu_id != -1
            else list(
                set(model.module.parameters()).difference(
                    set(model.module.model.embedding.parameters())
                )
            )
        )
    },
    {
        "params": (
            model.model.embedding.parameters()
            if args.gpu_id != -1
            else model.module.model.embedding.parameters()
        ),
        "lr": float(args.lr) * 1,
    },
]
if list(criterion.parameters()):
    param_groups.append(
        {"params": criterion.parameters(), "lr": float(args.lr) * float(args.lr_ratio)}
    )

# Optimizer Setting
if args.optimizer == "sgd":
    opt = torch.optim.SGD(
        param_groups,
        lr=float(args.lr),
        weight_decay=args.weight_decay,
        momentum=0.9,
        nesterov=True,
    )
elif args.optimizer == "adam":
    opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)
elif args.optimizer == "rmsprop":
    opt = torch.optim.RMSprop(
        param_groups,
        lr=float(args.lr),
        alpha=0.9,
        weight_decay=args.weight_decay,
        momentum=0.9,
    )
elif args.optimizer == "adamw":
    opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.StepLR(
    opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma
)

print("Training parameters: {}".format(vars(args)))
print("Training for {} epochs.".format(args.nb_epochs))
losses_list = []
best_metrics = {args.metric_to_monitor: 0}
best_epoch = 0

for epoch in range(0, args.nb_epochs):
    model.train()
    bn_freeze = args.bn_freeze
    if bn_freeze:
        modules = model.model.modules() if args.gpu_id != -1 else model.module.model.modules()
        for m in modules:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    losses_per_epoch = []

    # Warmup: Train only new params, helps stabilize learning.
    if args.warm > 0:
        if args.gpu_id != -1:
            unfreeze_model_param = list(model.model.embedding.parameters()) + list(
                criterion.parameters()
            )
        else:
            unfreeze_model_param = list(model.module.model.embedding.parameters()) + list(
                criterion.parameters()
            )

        if epoch == 0:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = False
        if epoch == args.warm:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = True

    pbar = tqdm(enumerate(dl_tr))

    for batch_idx, (x, y) in pbar:
        m = model(x.squeeze().cuda())
        loss = criterion(m, y.squeeze().cuda())

        opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        if args.loss == "Proxy_Anchor":
            torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)

        losses_per_epoch.append(loss.data.cpu().numpy())
        opt.step()

        pbar.set_description(
            "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                epoch,
                batch_idx + 1,
                len(dl_tr),
                100.0 * batch_idx / len(dl_tr),
                loss.item(),
            )
        )

    losses_list.append(np.mean(losses_per_epoch))
    wandb.log({"loss": losses_list[-1]}, step=epoch)
    scheduler.step()

    if epoch >= 0:
        with torch.no_grad():
            print("**Evaluating...**")
            if args.dataset == "Inshop":
                metrics = evaluate_cos_Inshop(model, dl_query, dl_gallery)
            elif args.dataset != "SOP":
                metrics = evaluate_cos(model, dl_ev)
            else:
                metrics = evaluate_cos_SOP(model, dl_ev)

        # Logging Evaluation Score
        for k, v in metrics.items():
            wandb.log({k: v}, step=epoch)

        # Best model save
        if best_metrics[args.metric_to_monitor] < metrics[args.metric_to_monitor]:
            counter = 0
            best_metrics = metrics
            best_epoch = epoch
            if not os.path.exists("{}".format(LOG_DIR)):
                os.makedirs("{}".format(LOG_DIR))
            torch.save(
                {"model_state_dict": model.state_dict()},
                "{}/{}_{}_best.pth".format(LOG_DIR, args.dataset, args.model),
            )
            with open(
                "{}/{}_{}_best_results.txt".format(LOG_DIR, args.dataset, args.model),
                "w",
            ) as f:
                f.write("Best Epoch: {}\n".format(best_epoch))
                for k, v in best_metrics.items():
                    f.write("{}: {}\n".format(k, v))
        else:
            counter += 1
            if args.patience != -1:
                print("Early stopping counter: {}".format(counter))
                if counter > args.patience:
                    print("Early stopping at epoch {}".format(epoch))
                break
