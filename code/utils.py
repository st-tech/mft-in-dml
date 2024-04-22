# Adapted from https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/tree/master/code/utils.py
import torch
from tqdm import tqdm
import torch.nn.functional as F
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t, y in zip(T, Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1.0 * len(T))


def predict_batchwise(model, dataloader):
    model_is_training = model.training
    model.eval()

    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = model(J.cuda())

                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training)  # revert to previous training state

    return [torch.stack(A[i]) for i in range(len(A))]


def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, *_ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T == class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean


def evaluate_cos(model, dataloader):

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # calculate accuracy in pytorch metric learning
    metrics = calc_pml_acc(query_X=X, query_T=T, gallery_X=X, gallery_T=T, ref_includes_query=True)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []

    cos_sim = F.linear(X, X).cpu()
    Y = T[cos_sim.topk(1 + K)[1][:, 1:]]
    Y = Y.float()

    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        metrics[f"R@{k}"] = r_at_k
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return metrics


def evaluate_cos_Inshop(model, query_dataloader, gallery_dataloader):
    # calculate embeddings with model and get targets
    query_X, query_T = predict_batchwise(model, query_dataloader)
    gallery_X, gallery_T = predict_batchwise(model, gallery_dataloader)

    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)

    # calculate accuracy in pytorch metric learning
    metrics = calc_pml_acc(
        query_X=query_X,
        query_T=query_T,
        gallery_X=gallery_X,
        gallery_T=gallery_T,
        ref_includes_query=False,
    )

    # get predictions by assigning nearest 8 neighbors with cosine
    cos_sim = F.linear(query_X, gallery_X)

    def recall_k(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0

        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]

            thresh = torch.max(pos_sim).item()

            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1

        return match_counter / m

    # calculate recall @ 1, 2, 4, 8
    for k in [1, 10, 20, 30, 40, 50]:
        r_at_k = recall_k(cos_sim, query_T, gallery_T, k)
        metrics[f"R@{k}"] = r_at_k
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return metrics


def evaluate_cos_SOP(model, dataloader):
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # calculate accuracy in pytorch metric learning
    metrics = calc_pml_acc(query_X=X, query_T=T, gallery_X=X, gallery_T=T, ref_includes_query=True)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 1000
    Y = []
    xs = []
    for x in X:
        if len(xs) < 10000:
            xs.append(x)
        else:
            xs.append(x)
            xs = torch.stack(xs, dim=0)
            cos_sim = F.linear(xs, X)
            y = T[cos_sim.topk(1 + K)[1][:, 1:]]
            Y.append(y.float().cpu())
            xs = []

    # Last Loop
    xs = torch.stack(xs, dim=0)
    cos_sim = F.linear(xs, X)
    y = T[cos_sim.topk(1 + K)[1][:, 1:]]
    Y.append(y.float().cpu())
    Y = torch.cat(Y, dim=0)

    # calculate recall @ 1, 2, 4, 8
    for k in [1, 10, 100, 1000]:
        r_at_k = calc_recall_at_k(T, Y, k)
        metrics[f"R@{k}"] = r_at_k
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    return metrics


def calc_pml_acc(query_X, query_T, gallery_X, gallery_T, ref_includes_query):
    acc = AccuracyCalculator()

    # calculate embeddings with model and get targets

    pml_acc = acc.get_accuracy(
        query=query_X.detach().cpu().numpy(),
        reference=gallery_X.detach().cpu().numpy(),
        query_labels=query_T,
        reference_labels=gallery_T,
        ref_includes_query=ref_includes_query,
        exclude=("AMI", "NMI"),
    )
    for k, v in pml_acc.items():
        print(f"{k}: {v}")
    return pml_acc
