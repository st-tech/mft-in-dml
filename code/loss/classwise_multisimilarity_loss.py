import torch


class ClassWiseMultiSimilarityLoss(torch.nn.Module):
    def __init__(self, alpha, beta, base):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base

    def get_target_mask(self, embeddings, labels):
        batch_size = labels.size(0)
        label_set = torch.unique(labels)
        mask = (labels.reshape(batch_size, 1) == label_set) * 1.0
        return mask.to(device=embeddings.device)

    def scale_embeddings(self, embeddings, eps=1e-12):
        return torch.nn.functional.normalize(embeddings, dim=1, eps=eps)

    def forward(
        self,
        embeddings,
        labels,
        indices_tuple=None,  # necessary for compatibility with pytorch-metric-learning
    ):
        dtype, device = embeddings.dtype, embeddings.device
        num_classes_in_batch = torch.unique(labels).shape[0]

        embeddings = self.scale_embeddings(embeddings)  # shape = (batch_size, embedding_size)
        target_mask = self.get_target_mask(
            embeddings=embeddings, labels=labels
        )  # shape = (batch_size, num_classes)
        mask = torch.eye(
            n=target_mask.shape[-1], device=device
        )  # shape = (num_classes, num_classes)

        target_mask_normalized = target_mask / torch.maximum(
            torch.sum(target_mask, dim=0), torch.ones(1, device=target_mask.device)
        )

        cos = embeddings @ embeddings.T  # shape = (batch_size, batch_size)
        pos_logit = torch.exp(-self.alpha * (cos - self.base))  # shape = (batch_size, batch_size)
        neg_logit = torch.exp(self.beta * (cos - self.base))  # shape = (batch_size, batch_size)

        pos = (
            target_mask_normalized.T @ pos_logit @ target_mask_normalized
        )  # shape = (num_classes, num_classes)
        neg = (
            target_mask_normalized.T @ neg_logit @ target_mask_normalized
        )  # shape = (num_classes, num_classes)

        loss = torch.sum(torch.log(1 + pos * mask / 2)) / (
            self.alpha * num_classes_in_batch
        ) + torch.sum(torch.log(1 + neg * (1 - mask))) / (2 * self.beta * num_classes_in_batch)

        return loss
