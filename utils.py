import torch.nn.functional as F

def loss_fn(i, prediction, label):
    lamda = [1., 1., 1., 1., 1., 1.]
    cost = F.binary_cross_entropy_with_logits(prediction, label)  # [20, 256, 256] [20, 256, 256] type == Tensor
    return cost * lamda[i]
