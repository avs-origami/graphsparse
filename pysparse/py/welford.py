import torch

class WelfordNorm:
    def __init__(self, shape, eps=1e-8):
        """
        Normalization using Welford's algorithm.
        Can be used for both state and reward normalization.
        """
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.M2 = torch.zeros(shape, dtype=torch.float32)
        self.count = 0
        self.eps = eps
        self.training = True # Only update the normalizer when training is True.

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def manual_load(self, mean, M2, count):
        self.mean.copy_(mean)
        self.M2.copy_(M2)
        self.count = count

    def update(self, x):
        """
        Update running statistics with a new sample x using Welford's algorithm.
        """
        if self.count == 0:
            # First sample: initialize mean and zero-out M2.
            self.mean.copy_(x)
            self.M2.zero_()
            self.count = 1
        else:
            self.count += 1
            delta = x - self.mean
            self.mean.add_(delta / self.count)
            delta2 = x - self.mean
            self.M2.add_(delta * delta2)

    def var(self):
        if self.count < 2:
            # Not enough samples: return a tensor of ones with the same shape as mean
            return torch.ones_like(self.mean)
        else:
            return self.M2 / (self.count - 1)

    def std(self):
        return torch.sqrt(self.var()) + self.eps

    def norm(self, x) -> torch.Tensor :
        """
        Normalizes the sample x using the running mean and standard deviation.
        - x (torch.Tensor or array-like): The sample to normalize.
        - update (bool): If True, update the running statistics with x.
        - Returns the normalized sample.
        """
        if self.training:
            self.update(x)
        return (x - self.mean) / self.std()