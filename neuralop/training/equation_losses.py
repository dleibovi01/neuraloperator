import numpy as np
import torch
import scipy.io
from neuralop.training.data_losses import LpLoss
from neuralop.training.FC_utils import FC2D
from pathlib import Path
        
        

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
        
        
class DarcyEqnFCLoss(object):

    def __init__(self, d = 5, C = 25, nx = 100, ny = 100, domain_length_x=1, domain_length_y=1):
        self.FC = FC2D(d, C, nx, ny, domain_length_x, domain_length_y)
        super().__init__()
        
    def FC_diff(self, a, u):

        # remove extra channel dimensions
        u = u[:, 0, :, :]
        a = a[:, 0, :, :]
        
        u = torch.squeeze(u)
        a = torch.squeeze(a)
            

        # compute derivatives along the x-direction
        ux = self.FC.diff_x(u)	
             
        # compute derivatives along the y-direction
        uy = self.FC_diff_y(u)
        
        a_ux = a * ux
        a_uy = a * uy

        # compute derivatives along the x-direction
        a_uxx = self.FC.diff_x(a_ux)
        
        # compute derivatives along the y-direction
        a_uyy = self.FC.diff_y(a_uy)
        
        
        left_hand_side =  -(a_uxx + a_uyy)

        # compute the Lp loss of the left and right hand sides of the Darcy Flow equation
        forcing_fn = torch.ones(left_hand_side.shape, device=u.device)
        lploss = LpLoss(d=2, reductions='mean') # todo: size_average=True
        return lploss.rel(left_hand_side, forcing_fn)        
          
        
    def __call__(self, a, u, _):
        return self.FC_diff(a, u)        
        
        
        
        

