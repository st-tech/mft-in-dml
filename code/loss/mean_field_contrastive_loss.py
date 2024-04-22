import torch
import torch.nn.functional as F

from .base_mean_field_loss import BaseMeanFieldLoss


class MeanFieldContrastiveLoss(BaseMeanFieldLoss):
    def __init__(
        self,
        num_classes,
        embedding_size,
        pos_margin,
        neg_margin,
        mf_reg,
        mf_power,
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            **kwargs,
        )
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.mf_reg = mf_reg
        self.uniform = True
        self.mf_power = mf_power

    def get_loss(self, embeddings, labels):
        mask = self.get_target_mask(embeddings=embeddings, labels=labels)
        distance_from_class_vec = self.get_distance_from_class_vec(
            embeddings=embeddings
        )  # 1 - emb @ W >= 0 on "Sphere"
        pos = F.relu(distance_from_class_vec - self.pos_margin) * mask
        neg = F.relu(self.neg_margin - distance_from_class_vec) * (1 - mask)
        loss = torch.mean(torch.sum(pos + neg, dim=1))
        return loss

    def get_reg(self):
        class_matrix = self.get_class_distance_matrix()
        mask = torch.eye(self.num_classes, device=class_matrix.device)
        reg = self.mf_reg * torch.mean(
            torch.sum(
                torch.pow(
                    F.relu(self.neg_margin - class_matrix) * (1 - mask),
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

        embeddings = self.scale_embeddings(embeddings)

        loss = self.get_loss(embeddings=embeddings, labels=labels)
        reg = self.get_reg()
        total_loss = loss + reg

        return total_loss
