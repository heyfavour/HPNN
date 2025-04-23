import torch
from torch_scatter import scatter


class CosineCutoff(torch.nn.Module):

    def __init__(self, cutoff: float):
        super(CosineCutoff, self).__init__()
        self.cutoff = cutoff

    def forward(self, dist):
        # 0.5*(cos(pi*r/cutoff)+1)
        soft_cutoff = 0.5 * (torch.cos(dist * torch.pi / self.cutoff) + 1.0)
        soft_cutoff = soft_cutoff * (dist < self.cutoff).float()  # radius_graph 已经做过判断,理论上不需要判断dist<cutoff
        return soft_cutoff


class Guass_Basis(torch.nn.Module):

    def __init__(self, num_rbf=40, node_dim=128, pair_dim=64, cutoff=7.0, num_layers=3):
        super().__init__()
        offset = torch.linspace(0, cutoff, num_rbf)
        self.num_rbf = num_rbf
        self.node_dim = node_dim
        self.cutoff = cutoff
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2  # 是一个具体的数值
        self.radical_layers = torch.nn.Sequential(
            torch.nn.Linear(num_rbf, node_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(node_dim, pair_dim * num_layers),
        )
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        gaussian_rbf = torch.exp(self.coeff * torch.pow(dist, 2))
        rbf = self.radical_layers(gaussian_rbf)
        return rbf




class SphConv2(torch.nn.Module):
    # 一层sph如果层数不同调用不同的sphconv,使用循环判断影响执行速度
    # len(sph) == 2 [6 0] [2 0] [1 0]
    def __init__(self, node_dim,pair_dim,sph,dropout):
        #######################################################
        # init
        super().__init__()
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.sph = sph
        assert len(sph) == 2
        assert sph[-1] == 0
        #######################################################
        self.x_down = torch.nn.Linear(node_dim, pair_dim)
        
        self.x_up0 = torch.nn.Linear(pair_dim,pair_dim*2)
        self.x_uph = torch.nn.Linear(pair_dim,pair_dim*2,bias=False)

        self.node_layer = torch.nn.Sequential(
            torch.nn.Linear(pair_dim*4, node_dim),
            torch.nn.LayerNorm(node_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self,x,j,i,rbf_r,factor_r,sph_v):
        _x  = self.x_down(x)
        x_i  = _x[i]
        x_j  = _x[j]
        #######################################################
        l0_pair = x_j*rbf_r*factor_r
        lh_pair = l0_pair.unsqueeze(1)*sph_v[0]
        l0_node = self.x_up0(_x*scatter(l0_pair,i,dim=0))
        lh_node = self.x_uph(_x.unsqueeze(1)*scatter(lh_pair,i,dim=0))
        lh_node = torch.sum(torch.square(lh_node),dim=1)


        node = torch.cat([l0_node,lh_node],dim=-1)
        node = self.node_layer(node) + x
        return node


class SphConv3(torch.nn.Module):
    # 一层sph如果层数不同调用不同的sphconv,使用循环判断影响执行速度
    # len(sph) == 3 [2 1 0] [2 1 0] [2 1 0]
    def __init__(self, node_dim,pair_dim,sph,dropout):
        #######################################################
        # init
        super().__init__()
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.sph = sph
        assert len(sph) == 3
        assert sph[-1] == 0
        #######################################################
        self.x_down = torch.nn.Linear(node_dim, pair_dim)
        
        self.x_up0 = torch.nn.Linear(pair_dim,pair_dim*2)
        self.x_up1 = torch.nn.Linear(pair_dim,pair_dim*2,bias=False)
        self.x_uph = torch.nn.Linear(pair_dim,pair_dim*2,bias=False)

        self.node_layer = torch.nn.Sequential(
            torch.nn.Linear(pair_dim*6, node_dim),
            torch.nn.LayerNorm(node_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self,x,j,i,rbf_r,factor_r,sph_v):
        _x  = self.x_down(x)
        x_j  = _x[j]
        #######################################################
        l0_pair = x_j*rbf_r*factor_r
        l1_pair = l0_pair.unsqueeze(1)*sph_v[1]
        lh_pair = l0_pair.unsqueeze(1)*sph_v[0]


        l0_node = self.x_up0(_x*scatter(l0_pair,i,dim=0))
        l1_node = self.x_up1(_x.unsqueeze(1)*scatter(l1_pair,i,dim=0))
        lh_node = self.x_uph(_x.unsqueeze(1)*scatter(lh_pair,i,dim=0))

        l1_node = torch.sum(torch.square(l1_node),dim=1) 
        lh_node = torch.sum(torch.square(lh_node),dim=1)

        node = torch.cat([l0_node,l1_node,lh_node],dim=-1)
        node = self.node_layer(node) + x
        return node


def InitSphConv(node_dim,pair_dim,sph,dropout):
    if len(sph) == 2:
        return SphConv2(node_dim,pair_dim,sph,dropout)
    elif len(sph) == 3:
        return SphConv3(node_dim,pair_dim,sph,dropout)
    else:
        raise Exception(f"unkonwn sph:{sph}")

if __name__ == '__main__':
    pass
