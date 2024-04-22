import torch
from .base_mean_field_loss import BaseMeanFieldLoss


class MeanFieldClassWiseMultiSimilarityLoss(BaseMeanFieldLoss):
    def __init__(
        self,
        num_classes,
        embedding_size,
        alpha,
        beta,
        base,
        mf_reg,
        mf_power,
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            **kwargs,
        )
        self.alpha = alpha
        self.beta = beta if not beta is None else alpha
        self.base = base
        self.mf_reg = mf_reg
        self.mf_power = mf_power

    def get_loss(self, embeddings, labels):
        embeddings = self.scale_embeddings(embeddings)
        num_classes_in_batch = torch.unique(labels).shape[0]
        distance_from_class_vec = self.get_distance_from_class_vec(
            embeddings=embeddings
        )  # 1 - emb @ W >= 0 on "S" # shape = (batch_size, num_classes)

        target_mask = self.get_target_mask(
            embeddings=embeddings, labels=labels
        )  # shape = (batch_size, num_classes)
        mask = torch.eye(
            n=self.num_classes, device=embeddings.device
        )  # shape = (num_classes, num_classes)

        target_mask_normalized = target_mask / torch.maximum(
            torch.sum(target_mask, dim=0), torch.ones(1, device=target_mask.device)
        )

        pos_logit = torch.exp(self.alpha * (distance_from_class_vec + self.base - 1))
        neg_logit = torch.exp(-self.beta * (distance_from_class_vec + self.base - 1))
        pos = target_mask_normalized.T @ pos_logit
        neg = target_mask_normalized.T @ neg_logit
        neg = neg + neg.T

        loss = torch.sum(torch.log(1 + pos * mask)) / (
            self.alpha * num_classes_in_batch
        ) + torch.sum(torch.log(1 + neg * (1 - mask))) / (2 * self.beta * num_classes_in_batch)
        return loss

    def get_reg(self):
        class_matrix = self.get_class_distance_matrix()
        mask = torch.eye(self.num_classes, device=class_matrix.device)
        reg = self.mf_reg * torch.mean(
            torch.sum(
                torch.pow(
                    torch.nn.Softplus(self.beta)(1 - self.base - class_matrix) * (1 - mask),
                    self.mf_power,
                ),
                dim=1,
            )
        )
        return reg

    def forward(
        self,
        embeddings,
        labels,
        indices_tuple=None,  # necessary for compatibility with pytorch-metric-learning
    ):
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)

        loss = self.get_loss(embeddings=embeddings, labels=labels)
        reg = self.get_reg()

        total_loss = loss + reg
        return total_loss
