import torch


def to_dtype(x, tensor=None, dtype=None):
    # Utility functions implemented in pytorch-metric-learning>=1
    # but not in pytorch-metric-learning==0.9.88
    if not torch.is_autocast_enabled():
        dt = dtype if dtype is not None else tensor.dtype
        if x.dtype != dt:
            x = x.type(dt)
    return x


def to_device(x, tensor=None, device=None, dtype=None):
    # Utility functions implemented in pytorch-metric-learning>=1
    # but not in pytorch-metric-learning==0.9.88
    dv = device if device is not None else tensor.device
    if x.device != dv:
        x = x.to(dv)
    if dtype is not None:
        x = to_dtype(x, dtype=dtype)
    return x


class BaseMeanFieldLoss(
    torch.nn.Module,
):
    def __init__(
        self,
        num_classes,
        embedding_size,
        **kwargs,
    ):
        super().__init__(
            # weight_regularizer, weight_reg_weight, # pml>=1
            **kwargs
        )
        self.num_classes = num_classes
        self.W = torch.nn.Parameter(torch.Tensor(embedding_size, num_classes))
        torch.nn.init.normal_(self.W)  # torch.nn.init.kaiming_normal_(self.W)

    def cast_types(self, dtype, device):
        self.W.data = to_device(self.W.data, device=device, dtype=dtype)

    def get_class_distribution(self, embeddings, labels):
        batch_size = labels.size(0)
        class_distribution = torch.zeros(self.num_classes, device=embeddings.device)
        unique_labels, counts = torch.unique(labels, return_counts=True)
        class_distribution[unique_labels.type(torch.long)] = counts / batch_size
        return class_distribution

    def get_target_mask(self, embeddings, labels):
        batch_size = labels.size(0)
        mask = torch.zeros(
            batch_size,
            self.num_classes,
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        mask[torch.arange(batch_size), labels] = 1
        return mask

    def get_target_class_vec(self, embeddings, labels):
        mask = self.get_target_mask(embeddings, labels)
        return mask @ self.W.t()

    def scale_embeddings(self, embeddings, eps=1e-12):
        return torch.nn.functional.normalize(embeddings, dim=1, eps=eps)

    def get_class_distance_matrix(self):
        W_scaled = torch.nn.functional.normalize(self.W, dim=0)
        return 1 - torch.matmul(W_scaled.t(), W_scaled)

    def get_distance_from_class_vec(self, embeddings):
        return 1 - embeddings @ torch.nn.functional.normalize(self.W, dim=0)
