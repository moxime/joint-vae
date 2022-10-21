import torch

shape = (4096, 3, 32, 32)

x = torch.randn(*shape)


for _ in ('cuda', 'cpu', 'cuda:0', 'cuda:1'):

    print(_, x.to(_).device)
