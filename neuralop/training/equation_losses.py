import numpy as np
import torch
import scipy.io
from neuralop.training.data_losses import LpLoss
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
    def __init__(self):
        super().__init__()
        
    def FC_diff(a, u, A = torch.from_numpy(scipy.io.loadmat(Path(__file__).resolve().parent.joinpath("FC_data/A_d5_C25.mat"))['A']).double(), Q =    torch.from_numpy(scipy.io.loadmat(Path(__file__).resolve().parent.joinpath("FC_data/Q_d5_C25.mat"))['Q']).double(), domain_length_x=1, domain_length_y=1, d=5, C=25):


        # remove extra channel dimensions
        u = u[:, 0, :, :]
        a = a[:, 0, :, :]
        
        u = torch.squeeze(u)
        a = torch.squeeze(a)
            
	# Grid data
        ny = u.size(1)
        nx = u.size(2)

        hx = domain_length_x / (nx - 1)
        hy = domain_length_y / (ny - 1)

        fourPtsx = nx + C
        prdx = fourPtsx * hx
        fourPtsy = ny + C
        prdy = fourPtsy * hy	

        if fourPtsx % 2 == 0:
                k_max = int(fourPtsx/ 2)
                k_x = torch.cat((torch.arange(start = 0, end = k_max + 1, step = 1, device = 'cpu'),
                    torch.arange(start = - k_max + 1, end = 0, step = 1, device = 'cpu')), 0)
        else:
                k_max = int((fourPtsx - 1) / 2)
                k_x = torch.cat((torch.arange(start = 0, end = k_max + 1, step = 1, device = 'cpu'),
                    torch.arange(start = - k_max, end = 0, step = 1, device = 'cpu')), 0)

        
        if fourPtsy % 2 == 0:
                k_max = int(fourPtsy/ 2)
                k_y = torch.cat((torch.arange(start = 0, end = k_max + 1, step = 1, device = 'cpu'),
                    torch.arange(start = - k_max + 1, end = 0, step = 1, device = 'cpu')), 0).reshape(fourPtsy, 1).repeat(ny, 1)
        else:
                k_max = int((fourPtsy - 1) / 2)
                k_y = torch.cat((torch.arange(start = 0, end = k_max + 1, step = 1, device = 'cpu'),
                    torch.arange(start = - k_max, end = 0, step = 1, device = 'cpu')), 0).reshape(fourPtsy, 1).repeat(1, ny)	                        
	
	     
        der_coeffsx = 1j * 2.0 * np.pi / prdx * k_x
        der_coeffsy = 1j * 2.0 * np.pi / prdy * k_y


        # compute derivatives along the x-direction
        y1 = torch.einsum("hik,jk->hij", torch.einsum("hik,kj->hij", u[:, :, -d:], Q), A)
        y2 = torch.flip(torch.einsum("hik,jk->hij", torch.einsum("hik,kj->hij", torch.flip(u[:, :, :d], dims=(2,)), Q), A), dims=(2,))
        ucont = torch.cat([u,y1+y2], dim=2)
        uhat = torch.fft.fft(ucont, dim=2)
        uder = torch.fft.ifft(uhat * der_coeffsx).real
        # ux = uder[:, 1:-1, 1:nx-1]	
        ux = uder[:, :, :nx]	
        
     
        # compute derivatives along the y-direction
        y1 = torch.einsum("ikl,jk->ijl", torch.einsum("ikl,kj->ijl", u[:, -d:, :], Q), A)        
        y2 = torch.flip(torch.einsum("ikl,jk->ijl", torch.einsum("ikl,kj->ijl", torch.flip(u[:, :d, :], dims=(1,)), Q), A), dims=(1,))
        ucont = torch.cat([u,y1+y2], dim=1)
        uhat = torch.fft.fft(ucont, dim=1)
        uder = torch.fft.ifft(uhat * der_coeffsy, dim=1).real
        # uy = uder[:, 1:ny-1, 1:-1]	
        uy = uder[:, :ny, :]
        

        # a = a[:, 1:-1, 1:-1]
        a_ux = a * ux
        a_uy = a * uy


        # compute derivatives along the x-direction
        y1 = torch.einsum("hik,jk->hij", torch.einsum("hik,kj->hij", a_ux[:, :, -d:], Q), A)
        y2 = torch.flip(torch.einsum("hik,jk->hij", torch.einsum("hik,kj->hij", torch.flip(a_ux[:, :, :d], dims=(2,)), Q), A), dims=(2,))
        ucont = torch.cat([a_ux,y1+y2], dim=2)
        uhat = torch.fft.fft(ucont, dim=2)
        uder = torch.fft.ifft(uhat * der_coeffsx).real
        # a_uxx = uder[:, 1:-1, 1:nx-1]	
        a_uxx = uder[:, :, :nx]	
        
     
        # compute derivatives along the y-direction
        y1 = torch.einsum("ikl,jk->ijl", torch.einsum("ikl,kj->ijl", a_uy[:, -d:, :], Q), A)        
        y2 = torch.flip(torch.einsum("ikl,jk->ijl", torch.einsum("ikl,kj->ijl", torch.flip(a_uy[:, :d, :], dims=(1,)), Q), A), dims=(1,))
        ucont = torch.cat([a_uy,y1+y2], dim=1)
        uhat = torch.fft.fft(ucont, dim=1)
        uder = torch.fft.ifft(uhat * der_coeffsy, dim=1).real
        # a_uyy = uder[:, 1:ny-1, 1:-1]	
        a_uyy = uder[:, :ny, :]	


        left_hand_side =  -(a_uxx + a_uyy)


        left_hand_side =  -(a_uxx + a_uyy)	
        
        # compute the Lp loss of the left and right hand sides of the Darcy Flow equation
        forcing_fn = torch.ones(left_hand_side.shape, device=u.device)
        lploss = LpLoss(d=2, reductions='mean') # todo: size_average=True
        return lploss.rel(left_hand_side, forcing_fn)        
          
        
    def __call__(self, a, u, _):
        return self.FC_diff(a, u)        
        
        
        
        

