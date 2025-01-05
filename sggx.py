from utils import *


class FiberLikeSGGX:
    # input: w(..., 3)
    def sigma(self, w, alpha):
        return torch.sqrt(w[..., [0]] ** 2 + w[..., [1]] ** 2 + (w[..., [2]] * alpha) ** 2)

    def sigmaT(self, w, alpha, density=1.0):
        return self.sigma(w, alpha) * density

    def D(self, w, alpha):
        s2 = w[..., [0]] ** 2 + w[..., [1]] ** 2 + (w[..., [2]] / alpha) ** 2
        result = 1 / (PI * alpha * torch.square(s2))
        return result

    def eval(self, wi, half, alpha):
        return self.D(half, alpha) * 0.25 / self.sigma(wi, alpha)


class SurfaceLikeSGGX:
    def sigma(self, w, alpha):
        return torch.sqrt((w[..., [0]] * alpha) ** 2 + (w[..., [1]] * alpha) ** 2 + w[..., [2]] ** 2)

    def sigmaT(self, w, alpha, density=1.0):
        return self.sigma(w, alpha) * density

    def D(self, w, alpha):
        s2 = (w[..., [0]] / alpha) ** 2 + (w[..., [1]] / alpha) ** 2 + w[..., [2]] ** 2
        result = 1 / (PI * alpha * alpha * torch.square(s2))
        return result

    def eval(self, wi, half, alpha):
        return self.D(half, alpha) * 0.25 / self.sigma(wi, alpha)


