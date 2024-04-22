# Adapted from https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/tree/master/code/evaluate.py
import os
import sys
import argparse
import torch
import random
import numpy as np

import dataset
from dataset.Inshop import Inshop_Dataset
from net.resnet import *
from net.googlenet import *
from net.bn_inception import *
from utils import evaluate_cos, evaluate_cos_SOP, evaluate_cos_Inshop


parser = argparse.ArgumentParser(
    description="Official implementation of `Mean Field Theory in Deep Metric Learning` "
    + "Our code is modified from `https://github.com/dichotomies/proxy-nca` "
    + "and `https://github.com/tjddus9597/Proxy-Anchor-CVPR2020`."
)
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
parser.add_argument("--gpu-id", default=0, type=int, help="ID of GPU that is used for training.")
parser.add_argument(
    "--workers",
    default=4,
    type=int,
    dest="nb_workers",
    help="Number of workers for dataloader.",
)
parser.add_argument("--model", default="bn_inception", help="Model for training")
parser.add_argument("--l2-norm", default=1, type=int, help="L2 normlization")
parser.add_argument("--seed", default=1, type=int, help="Seed for random number generator")
parser.add_argument("--resume", default="", help="Path of resuming model")
parser.add_argument("--remark", default="", help="Any reamrk")

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  # set random seed for all gpus

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Data Root Directory
os.chdir("../data/")
data_root = os.getcwd()

# Dataset Loader and Sampler
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

# Backbone Model
if args.model.find("googlenet") + 1:
    model = googlenet(
        embedding_size=args.sz_embedding,
        pretrained=True,
        is_norm=args.l2_norm,
        bn_freeze=1,
    )
elif args.model.find("bn_inception") + 1:
    model = bn_inception(
        embedding_size=args.sz_embedding,
        pretrained=True,
        is_norm=args.l2_norm,
        bn_freeze=1,
    )
elif args.model.find("resnet18") + 1:
    model = Resnet18(
        embedding_size=args.sz_embedding,
        pretrained=True,
        is_norm=args.l2_norm,
        bn_freeze=1,
    )
elif args.model.find("resnet50") + 1:
    model = Resnet50(
        embedding_size=args.sz_embedding,
        pretrained=True,
        is_norm=args.l2_norm,
        bn_freeze=1,
    )
elif args.model.find("resnet101") + 1:
    model = Resnet101(
        embedding_size=args.sz_embedding,
        pretrained=True,
        is_norm=args.l2_norm,
        bn_freeze=1,
    )
model = model.cuda()

if args.gpu_id == -1:
    model = nn.DataParallel(model)

if os.path.isfile(args.resume):
    print("=> loading checkpoint {}".format(args.resume))
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    print("=> No checkpoint found at {}".format(args.resume))
    sys.exit(0)

with torch.no_grad():
    print("**Evaluating...**")
    if args.dataset == "Inshop":
        Recalls = evaluate_cos_Inshop(model, dl_query, dl_gallery)

    elif args.dataset != "SOP":
        Recalls = evaluate_cos(model, dl_ev)

    else:
        Recalls = evaluate_cos_SOP(model, dl_ev)
