import torch
import torch.nn as nn
import numpy as np

class SphericalHarmonicsSolver(nn.Module):
    def __init__(self,degree=3,init_regs=(0.001,0.005,0.05,0.1)):
        super().__init__()
        assert(degree<=4)
        self.func = self.get_coefficient_func()[:degree+1]
        regs = [np.zeros([1]), np.ones([1 * 2 + 1]) * init_regs[0], np.ones([2 * 2 + 1]) * init_regs[1], np.ones([3 * 2 + 1]) * init_regs[2], np.ones([4 * 2 + 1]) * init_regs[3]]
        regs = torch.from_numpy(np.concatenate(regs[:degree+1],0).astype(np.float32))
        self.register_buffer('regs',regs)

    @staticmethod
    def get_coefficient_func():
        l0=lambda x,y,z: torch.ones_like(x).unsqueeze(-1)
        l1=lambda x,y,z: torch.stack([x,y,z],-1)
        l2=lambda x,y,z: torch.stack([x*y,y*z,-x**2-y**2+2*z**2,z*x,x**2-y**2],-1)
        l3=lambda x,y,z: torch.stack([(3*x**2-y**2)*y,x*y*z,y*(4*z**2-x**2-y**2), z*(2*z**2-3*x**2-3*y**2),x*(4*z**2-x**2-y**2), (x**2-y**2)*z,(x**2-3*y**2)*x],-1)
        l4=lambda x,y,z: torch.stack([x*y*(x**2-y**2),
                                      (3*x**2-y**2)*y*z,
                                      x*y*(7*z**2-1),
                                      y*z*(7*z**2-3),
                                      35*z**4-30*z**2+3,
                                      x*z*(7*z**2-3),
                                      (x**2-y**2)*(7*z**2-1),
                                      (x**2-3*y**2)*x*z,
                                      x**2*(x**2-3*y**2)-y**2*(3*x**2-y**2)
                                      ],-1)
        return [l0,l1,l2,l3,l4]

    def forward(self, directions, colors, weights, eps=1e-4):
        """
        :param directions: [b,n,3]
        :param colors:     [b,n,3]
        :param weights:    [b,n]
        :param eps:
        :return:
        """
        x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
        A = torch.cat([fn(x,y,z) for fn in self.func],-1) # [b,n,k]
        insufficient_mask = torch.sum(weights,1,keepdim=True) < eps
        weights = weights + insufficient_mask.float() * eps
        A_ = (A * weights.unsqueeze(-1)).permute(0,2,1)
        # ([b,k,n] @ [b,n,k]) @ ([b,k,n] @ [b,n,3])
        regs = self.regs
        inv_mat = A_ @ A + torch.diag(regs).unsqueeze(0)
        theta = torch.inverse(inv_mat) @ (A_ @ colors)
        return theta # b,k,3

    def predict(self, directions, theta):
        """
        :param directions: [b,n,3]
        :param theta:      [b,k,3]
        :return:
        """
        x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
        A = torch.cat([fn(x,y,z) for fn in self.func],-1) # [b,n,k]
        return A @ theta
