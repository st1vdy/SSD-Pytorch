from torch import Tensor
import torch

a = Tensor([[0,1],[2,3],[-1,5]])
b, c = a.max(dim=1)
print(b, c)