import numpy as np
import torch
import scipy.io
from neuralop.training.data_losses import LpLoss
from pathlib import Path
        
        

class FC2D(object):
    
    
    
    def __init__(self, d = 5, C = 25, nx = 100, ny = 100, domain_length_x=1, domain_length_y=1):
    
        self.d = d
        self.C = C
        self.A = torch.from_numpy(scipy.io.loadmat(Path(__file__).resolve().parent.joinpath("FC_data/A_d{d}_C{C}.mat"))['A']).double()
        self.Q = torch.from_numpy(scipy.io.loadmat(Path(__file__).resolve().parent.joinpath("FC_data/Q_d{d}_C{C}.mat"))['Q']).double()
        self.nx = nx
        self.ny = ny
        
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
        
        self.der_coeffsx = der_coeffsx
        self.der_coeffsy = der_coeffsy

        
    def diff_x(u):

        # compute derivatives along the x-direction
        y1 = torch.einsum("hik,jk->hij", torch.einsum("hik,kj->hij", u[:, :, -self.d:], self.Q), self.A)
        y2 = torch.flip(torch.einsum("hik,jk->hij", torch.einsum("hik,kj->hij", torch.flip(u[:, :, :self.d], dims=(2,)), self.Q), self.A), dims=(2,))
        ucont = torch.cat([u,y1+y2], dim=2)
        uhat = torch.fft.fft(ucont, dim=2)
        uder = torch.fft.ifft(uhat * self.der_coeffsx).real	
        ux = uder[:, :, :self.nx]	
        
        return ux
        
        
    def diff_y(u):

        # compute derivatives along the y-direction
        y1 = torch.einsum("ikl,jk->ijl", torch.einsum("ikl,kj->ijl", u[:, -self.d:, :], self.Q), self.A)        
        y2 = torch.flip(torch.einsum("ikl,jk->ijl", torch.einsum("ikl,kj->ijl", torch.flip(u[:, :self.d, :], dims=(1,)), self.Q), self.A), dims=(1,))
        ucont = torch.cat([u,y1+y2], dim=1)
        uhat = torch.fft.fft(ucont, dim=1)
        uder = torch.fft.ifft(uhat * self.der_coeffsy, dim=1).real
        uy = uder[:, :self.ny, :]
        
        return uy
        
    
        
        
        
        

