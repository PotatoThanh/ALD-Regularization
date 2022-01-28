import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from rbf import RBF

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def ald_pgd_loss(model,
                x_natural,
                y,
                device='cuda',
                n=4, # number of particles
                sigma=None, # sigma of RBF kernel
                step_size=0.003, #2/255 step size
                epsilon=0.031, #8/255 radius constraint
                perturb_steps=10, # number of iterations
                projecting=True, # projecto the ball constraint
                distance='l_inf', 
                x_min=0.0, 
                x_max=1.0):


    assert(distance == 'l_inf')
    assert(projecting is True)
    assert(x_max > x_min)

    kernel = RBF(n, sigma)
    
    batch_size, c, w, h = x_natural.size()

    model.eval()

    # random initialization
    x_particle = x_natural.repeat(n, 1, 1, 1)
    x_adv = Variable(x_particle.data, requires_grad=True)
    random_noise = torch.FloatTensor(*x_adv.shape).uniform_(-epsilon, epsilon).to(device)
    x_adv = Variable(x_adv.data + random_noise, requires_grad=True)

    for _ in range(perturb_steps):
        x_adv = x_adv.reshape(batch_size*n, -1)
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_ce = nn.CrossEntropyLoss(size_average=False)(model(x_adv.reshape(batch_size*n, c, w, h)),
                                                y.unsqueeze(-1).repeat(n, 1).squeeze(-1)) # Will not take average over batch 

        # print(loss_ce.item())
        score_func = torch.autograd.grad(loss_ce, [x_adv])[0]
        K_XX = kernel(x_adv, x_adv.detach())
        grad_K = -torch.autograd.grad(K_XX.sum(), x_adv)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / (batch_size*n)

        x_adv = x_adv.detach() + step_size * torch.sign(phi.detach())
        x_adv = x_adv.reshape(batch_size*n, c, w, h)
        if projecting:
            x_adv = torch.min(torch.max(x_adv, x_particle - epsilon), x_particle + epsilon)
        x_adv = torch.clamp(x_adv, x_min, x_max)

    # "#############"
    model.train()

    x_adv = Variable(torch.clamp(x_adv, x_min, x_max), requires_grad=False)

    # calculate robust loss
    adv_output = model(x_adv)
    loss_robust = F.cross_entropy(adv_output, y.unsqueeze(-1).repeat(n, 1).squeeze(-1), reduction='mean')
        
    return loss_robust, x_adv

