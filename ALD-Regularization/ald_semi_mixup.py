
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import contextlib
from svgd.rbf import RBF
from datetime import datetime

@contextlib.contextmanager
def disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def exp_rampup(rampup_length=30):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper

def l2_normalize(d):
    d_abs_max = torch.max(
        torch.abs(d.view(d.size(0),-1)), 1, keepdim=True)[0].view(
            d.size(0),1,1,1)
    d /= (1e-12 + d_abs_max)
    d /= torch.sqrt(1e-6 + torch.sum(
        torch.pow(d,2.0), tuple(range(1, len(d.size()))), keepdim=True))
    return d

def mse_with_softmax(logit1, logit2):
    assert logit1.size()==logit2.size()
    return F.mse_loss(F.softmax(logit1,1), F.softmax(logit2,1))

def gen_r_vadv(model, x, vlogits, n, sigma, xi, eps, niter):  
    kernel = RBF(n, sigma)
    batch_size, c, w, h = x.size()

    # perpare random unit tensor
    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = l2_normalize(d)
    # calc adversarial perturbation
    for _ in range(niter):
        d = d.reshape(batch_size, c, w, h)
        d.requires_grad_()
        with torch.enable_grad():
            rlogits = model(x + xi * d)
        adv_dist = mse_with_softmax(rlogits, vlogits)

        score_func = torch.autograd.grad(adv_dist.sum(), [d])[0]

        d = d.reshape(batch_size, -1)
        score_func = score_func.reshape(batch_size, -1)

        K_XX = kernel(d, d.detach())
        grad_K = -torch.autograd.grad(K_XX.sum(), d)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / (batch_size)
        phi = phi.reshape(batch_size, c, w, h)
        phi = l2_normalize(phi)
    return eps * phi.detach()

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    # if alpha > 0:
    lam = np.random.beta(alpha, alpha)
    # else:
    #     lam = 0.1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index, :]
    return mixed_x, y_a, y_b, lam

def ald_semi_mixup_loss(model,
                label_x, label_y, unlab_x,
                epoch #current epoch for exponential ramup
                n=4, # number of particles
                sigma=None, # sigma in RBF kernal
                xi=10.0,
                eps=0.0005,
                niter=1, # number of iterations
                alpha = 0.5 # alpha for mixup
                ):

    model.train(True)
    with torch.enable_grad():
        lbs, ubs = label_x.size(0), unlab_x.size(0)

        ##=== Semi-supervised Training ===
        ## local distributional smoothness (LDS)
        unlab_x = unlab_x.repeat(n, 1, 1, 1)
        
        unlab_outputs = model(unlab_x)
        with torch.no_grad():
            vlogits = unlab_outputs.clone().detach()

        with disable_tracking_bn_stats(model):
            r_vadv  = gen_r_vadv(model, unlab_x, vlogits, n, sigma, xi, eps, niter)
            x_adv = unlab_x + r_vadv

            x_adv, vlogits_a, vlogits_b, lam = mixup_data(x_adv, vlogits, alpha=alpha, use_cuda=True)

            rlogits = model(x_adv)
            loss_robust  = lam*mse_with_softmax(rlogits, vlogits_a) + (1.0-lam)*mse_with_softmax(rlogits, vlogits_b)
            loss_robust *= exp_rampup()(epoch)*beta

    return loss_robust, x_adv

