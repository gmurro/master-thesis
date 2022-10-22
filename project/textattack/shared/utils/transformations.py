import torch

class MinMaxScaler(object):
    """
    Transforms each channel to the range [0, 1].
    """
    def __call__(self, tensor):
        tensor = tensor.to(torch.float32)
        dist = (tensor.max() - tensor.min())
        dist[dist==0.] = 1.
        scale = 1.0 /  dist
        tensor.sub_(tensor.min()).mul_(scale)
        return tensor