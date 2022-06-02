import torch

x = torch.zeros(1)


for _ in ('cuda', 'cpu', 'cuda:0', 'cuda:1'):

    print(_, x.to(_).device)

