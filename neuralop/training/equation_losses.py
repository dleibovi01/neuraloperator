import numpy as np
import torch
from neuralop.training.data_losses import LpLoss

class DarcyEqnLoss(object):
    def __init__(self):
        super().__init__()

    def finite_difference(self, a, u, domain_length=1):
        # remove extra channel dimensions
        u = u[:, 0, :, :]
        a = a[:, 0, :, :]

        # compute the left hand side of the Darcy Flow equation
        # note: here we assume that the input is a regular grid
        n = u.size(1)
        dx = domain_length / (n - 1)
        dy = dx

        # todo: replace this with Jeff's central_diff_2d in the other branch
        ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
        uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

        a = a[:, 1:-1, 1:-1]
        a_ux = a * ux
        a_uy = a * uy

        # todo: replace this with Jeff's diff implementation in the other branch
        a_uxx = (a_ux[:, 2:, 1:-1] - a_ux[:, :-2, 1:-1]) / (2 * dx)
        a_uyy = (a_uy[:, 1:-1, 2:] - a_uy[:, 1:-1, :-2]) / (2 * dy)
        left_hand_side =  -(a_uxx + a_uyy)

        # compute the Lp loss of the left and right hand sides of the Darcy Flow equation
        forcing_fn = torch.ones(left_hand_side.shape, device=u.device)
        lploss = LpLoss(d=2, reductions='mean') # todo: size_average=True
        return lploss.rel(left_hand_side, forcing_fn)

    def __call__(self, a, u, _):
        return self.finite_difference(a, u)