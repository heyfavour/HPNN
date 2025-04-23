import e3nn
import torch

from torch.autograd import grad
from torch_scatter import scatter
from model_block import CosineCutoff, Guass_Basis,InitSphConv
from utils.common import radius_graph


class PTSDGraphNet(torch.nn.Module):
    def __init__(self, cutoff=5.0, node_dim=512, num_layers=3,pair_dim=128,num_rbf=48,sph=None,dropout=0.0,E_0=None):
        super().__init__()

        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.cutoff = cutoff  # 截断 5.0
        self.z_emb = torch.nn.Embedding(105, node_dim)
        self.E_emb = torch.nn.Embedding(105, 1)
        if E_0 is not None: self.E_emb.weight.data.copy_(E_0.view(105, 1)) 

        self.rbf_function = Guass_Basis(num_rbf=num_rbf, node_dim=node_dim, pair_dim=pair_dim, cutoff=cutoff,num_layers=num_layers)
        self.cutoff_factor = CosineCutoff(cutoff)
        self.num_layers = num_layers
        ################################################################################################################
        # 球谐 func
        self.sph,self.sph_function = sph,torch.nn.ModuleDict()
        assert len(sph) == num_layers
        _sph = set([j for i in sph for j in i if j!=0])
        for l in _sph:self.sph_function[str(l)] = e3nn.o3.SphericalHarmonics(l, True, "integral")
        ################################################################################################################
        # 球谐 layer
        self.sph_layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.sph_layers.append(InitSphConv(node_dim,pair_dim,sph[i],dropout))
        ################################################################################################################
        # 能量 layer
        self.energy_layer = torch.nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(node_dim, 1),
        )

    def forward(self, data):
        z, batch = data.z, data.batch
        edge_index = radius_graph(data, self.cutoff)
        j, i = edge_index
        ###############################################################stress
        # FP16 bmm误差积累较大
        with torch.amp.autocast('cuda',enabled=False):
            strain = torch.zeros_like(data.cell, requires_grad=True)
            eyes = torch.eye(3).to(strain.device).unsqueeze(0).repeat(data.num_graphs, 1, 1) + strain
            lattice = torch.bmm(data.cell.float(), eyes.float())
            cell = lattice[data.batch_j]
            image = data.image.unsqueeze(-2)
            shift = torch.bmm(image, cell).view(-1, 3)[data._j]
            volume = torch.linalg.det(lattice)
            #assert torch.allclose(volume, data.volume)
            scale = (1 / volume.view(-1, 1, 1))
            #assert torch.allclose(shift,data.shift[data._j],0.000001,0.0001)
            frac_cell = lattice[data.batch]
            unfrac_pos = torch.bmm(data.frac_pos.unsqueeze(1),frac_cell).squeeze(1)
            #assert torch.allclose(unfrac_pos, data.pos,0.000001,0.0001)
        ###############################################################球谐
        v_r = unfrac_pos[j] + shift - unfrac_pos[i]
        sph_v_r = {}
        for sph,sph_function in self.sph_function.items():
            sph_v_r[sph] = sph_function(v_r[:, [1, 2, 0]]).unsqueeze(2)  # 参数顺序为y z x
        ##############################################################距离
        s_r = v_r.norm(dim=-1).view(-1, 1)
        rbf_r = self.rbf_function(s_r)
        rbf_r = torch.split(rbf_r, self.pair_dim, dim=-1)
        factor_r = self.cutoff_factor(s_r)
        x = self.z_emb(z - 1)
        for num in range(self.num_layers):
            _sph = [sph_v_r[str(l)] for l in self.sph[num] if l!=0]
            x = self.sph_layers[num](x,j,i, rbf_r[num], factor_r, _sph)
        x = self.energy_layer(x) + self.E_emb(z - 1)
        energy = scatter(x, batch, dim=0)
        force, stress = grad(energy, [unfrac_pos, strain], grad_outputs=torch.ones_like(energy), create_graph=True, retain_graph=True)
        force, stress = -1 * force, scale * stress
        return energy, force, stress

if __name__ == '__main__':
    PTSDGraphNet()
