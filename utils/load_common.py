import math
import torch
import os, sys
import numpy as np
import pprint

from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import Data
from torch_cluster import knn
from torch_geometric.nn import radius



CACHE = {}
max_F = 100
check_radius = False
F_threshold = 1e-3

ELEMENT = {
    'H':    1, 'He':  2, 'Li':   3, 'Be':  4, 'B':    5, 'C':    6, 'N':   7, 'O':   8,
    'F':    9, 'Ne': 10, 'Na':  11, 'Mg': 12, 'Al':  13, 'Si':  14, 'P':  15, 'S':  16,
    'Cl':  17, 'Ar': 18, 'K':   19, 'Ca': 20, 'Sc':  21, 'Ti':  22, 'V':  23, 'Cr': 24,
    'Mn':  25, 'Fe': 26, 'Co':  27, 'Ni': 28, 'Cu':  29, 'Zn':  30, 'Ga': 31, 'Ge': 32,
    'As':  33, 'Se': 34, 'Br':  35, 'Kr': 36, 'Rb':  37, 'Sr':  38, 'Y':  39, 'Zr': 40,
    'Nb':  41, 'Mo': 42, 'Tc':  43, 'Ru': 44, 'Rh':  45, 'Pd':  46, 'Ag': 47, 'Cd': 48,
    'In':  49, 'Sn': 50, 'Sb':  51, 'Te': 52, 'I':   53, 'Xe':  54, 'Cs': 55, 'Ba': 56,
    'La':  57, 'Ce': 58, 'Pr':  59, 'Nd': 60, 'Pm':  61, 'Sm':  62, 'Eu': 63, 'Gd': 64,
    'Tb':  65, 'Dy': 66, 'Ho':  67, 'Er': 68, 'Tm':  69, 'Yb':  70, 'Lu': 71, 'Hf': 72,
    'Ta':  73, 'W':  74, 'Re':  75, 'Os': 76, 'Ir':  77, 'Pt':  78, 'Au': 79, 'Hg': 80,
    'Tl':  81, 'Pb': 82, 'Bi':  83, 'Po': 84, 'At':  85, 'Rn':  86, 'Fr': 87, 'Ra': 88,
    'Ac':  89, 'Th': 90, 'Pa':  91, 'U':  92, 'Np':  93, 'Pu':  94, 'Am': 95, 'Cm': 96,
    'Bk':  97, 'Cf': 98, 'Es':  99, 'Fm':100, 'Md': 101, 'No': 102, 'Lr':103, 'Rf':104,
    'Db': 105, 'Sg':106, 'Bh': 107, 'Hs':108, 'Mt': 109, 'Ds': 110, 'Rg':111, 'Cn':112,
    'Nh': 113, 'Fl':114, 'Mc':115,  'Lv':116, 'Ts': 117, 'Og':118
}

Eleradius = { 'H':  0.31,  'He':  0.28,  'Li':  1.28,  'Be':  0.96,   'B':  0.85,   'C':  0.76,   'N':  0.71,   'O':  0.66,
              'F':  0.57,  'Ne':  0.58,  'Na':  1.66,  'Mg':  1.41,  'Al':  1.21,  'Si':  1.11,   'P':  1.07,   'S':  1.05,
             'Cl':  1.02,  'Ar':  1.06,   'K':  2.03,  'Ca':  1.76,  'Sc':  1.70,  'Ti':  1.60,   'V':  1.53,  'Cr':  1.39,
             'Mn':  1.39,  'Fe':  1.32,  'Co':  1.26,  'Ni':  1.24,  'Cu':  1.32,  'Zn':  1.22,  'Ga':  1.22,  'Ge':  1.20,
             'As':  1.19,  'Se':  1.20,  'Br':  1.20,  'Kr':  1.16,  'Rb':  2.20,  'Sr':  1.95,   'Y':  1.90,  'Zr':  1.75,
             'Nb':  1.64,  'Mo':  1.54,  'Tc':  1.47,  'Ru':  1.46,  'Rh':  1.42,  'Pd':  1.39,  'Ag':  1.45,  'Cd':  1.44,
             'In':  1.42,  'Sn':  1.39,  'Sb':  1.39,  'Te':  1.38,   'I':  1.39,  'Xe':  1.40,  'Cs':  2.44,  'Ba':  2.15,
             'La':  2.07,  'Ce':  2.04,  'Pr':  2.03,  'Nd':  2.01,  'Pm':  1.99,  'Sm':  1.98,  'Eu':  1.98,  'Gd':  1.96,
             'Tb':  1.94,  'Dy':  1.92,  'Ho':  1.92,  'Er':  1.89,  'Tm':  1.90,  'Yb':  1.87,  'Lu':  1.87,  'Hf':  1.75,
             'Ta':  1.70,   'W':  1.62,  'Re':  1.51,  'Os':  1.44,  'Ir':  1.41,  'Pt':  1.36,  'Au':  1.36,  'Hg':  1.32,
             'Tl':  1.45,  'Pb':  1.46,  'Bi':  1.48,  'Po':  1.40,  'At':  1.50,  'Rn':  1.50,  'Fr':  2.60,  'Ra':  2.21,
             'Ac':  2.15,  'Th':  2.06,  'Pa':  2.00,   'U':  1.96,  'Np':  1.90,  'Pu':  1.87,  'Am':  1.80,  'Cm':  1.69,
             'Bk':  2.50,  'Cf':  2.50,  'Es':  2.50,  'Fm':  2.50,  'Md':  2.50,  'No':  2.50,  'Lr':  2.50,  'Rf':  2.50,
             'Db':  2.50,  'Sg':  2.50,  'Bh':  2.50,  'Hs':  2.50,  'Mt':  2.50,  'Ds':  2.50,  'Rg':  2.50,  'Cn':  2.50,
             'Nh':  2.50,  'Fl':  2.50,  'Mc':  2.50,  'Lv':  2.50,  'Ts':  2.50,  'Og':  2.50, }


ele_radius = torch.tensor([Eleradius[element] for element in sorted(ELEMENT, key=ELEMENT.get)])


def _min_zero(coor):  # 将cell过小的值置为0
    if abs(coor) < 1e-8: return 0
    return coor

def Lat(line):  # 将 a b c 三个角度转成 3*3的矩阵
    pbc = [float(l) for l in line]
    a, b, c = pbc[0:3]
    alpha, beta, gamma = [x * np.pi / 180.0 for x in pbc[3:]]

    bc2 = b ** 2 + c ** 2 - 2 * b * c * math.cos(alpha)
    h1 = _min_zero(a)
    h2 = _min_zero(b * math.cos(gamma))
    h3 = _min_zero(b * math.sin(gamma))
    h4 = _min_zero(c * math.cos(beta))
    h5 = _min_zero(((h2 - h4) ** 2 + h3 ** 2 + c ** 2 - h4 ** 2 - bc2) / (2 * h3))
    h6 = _min_zero(math.sqrt(c ** 2 - h4 ** 2 - h5 ** 2))
    lat = [[h1, 0., 0.], [h2, h3, 0.], [h4, h5, h6]]
    return lat

def check_positions_in_cell(pos, cell):  # 检查原子是否在晶胞内
    frac_coords = torch.linalg.solve(cell.T, pos.T).T
    is_inside = torch.all((frac_coords >= 0) & (frac_coords < 1)).item()
    return is_inside

def move_center_torch(pos, cell):  # 将原子移动到晶胞内
    frac_coords = torch.linalg.solve(cell.T, pos.T).T
    frac_coords = frac_coords % 1
    pos = frac_coords @ cell
    return pos

def check_pos(pos, cell):
    if not check_positions_in_cell(pos, cell): pos = move_center_torch(pos, cell)
    if not check_positions_in_cell(pos, cell): raise Exception("逻辑异常")
    return pos


def gen_cartesian_prod(_x, _y, _z,device):
    _x ,_y,_z= _x.item(),_y.item(),_z.item()

    if (_x, _y, _z) in CACHE: return CACHE[(_x, _y, _z)]
    x = torch.cat((torch.zeros(1), torch.arange(-1 * _x, 0), torch.arange(1, _x + 1)))
    y = torch.cat((torch.zeros(1), torch.arange(-1 * _y, 0), torch.arange(1, _y + 1)))
    z = torch.cat((torch.zeros(1), torch.arange(-1 * _z, 0), torch.arange(1, _z + 1)))
    cartesian_prod = torch.cartesian_prod(x, y, z).to(device)
    CACHE[(_x, _y, _z)] = cartesian_prod
    return cartesian_prod

def build_pbc(pos, cell, cutoff=5.0, device=None):  # 按照pos和晶胞 计算扩胞的基础数据
    inv_distances = torch.norm(cell.inverse().t(), dim=1)  # 轴的长度
    cell_xyz_repeats = torch.ceil(cutoff * inv_distances).long()
    #cell_xyz_repeats = torch.where(pbc, cell_xyz_repeats, cell_xyz_repeats.new_zeros(()))
    cell_num = torch.prod(2 * cell_xyz_repeats + 1, dim=0)  # 每个晶胞扩胞总计次数
    ##############################################################
    _x, _y, _z = cell_xyz_repeats
    # 解法1
    # _ = torch.mm(gen_cartesian_prod(_x, _y, _z), cell).repeat_interleave(pos.shape[0], 0)
    image = gen_cartesian_prod(_x, _y, _z,device).repeat_interleave(pos.shape[0], 0)
    # 解法2
    #_ = torch.bmm(image.unsqueeze(-2), cell.view(-1, 3, 3).repeat(image.shape[0], 1, 1)).view(-1, 3)
    # 解法3
    shift = image @ cell
    #assert torch.allclose(shift,_,0.000001,0.0001)
    pos_j = pos.repeat((cell_num, 1)) + shift  # j的坐标
    ##############################################################
    i_num = torch.tensor(pos.shape[0])
    j_num = torch.tensor(shift.shape[0])
    return image, shift, cell_num, i_num, j_num, pos_j


def check_loop(egde, node):# 检查node是否只有自环
    return egde.numel() == 1 and egde.item() == node

def check_healthy(data, cutoff, device=torch.device("cpu")):
    # 直接radius是i j radius_graph 是j i 
    #########################################################################nan校验
    if torch.isnan(data.energy).any().item():return False , "energy nan"
    if torch.isnan(data.force).any().item():return False , "force nan"
    if torch.isnan(data.stress).any().item():return False , "stress nan"
    if torch.any(data.force > max_F):return False ,"max force"
    #########################################################################force==[0,0,0]校验
    zero_v = torch.zeros((1, 3),device=device)
    exists = (data.force == zero_v).all(dim=1).any().item()  # 存在force = [0,0,0]
    if exists:return False, "force 0"
    ##########################################################################只跟自己连接
    # 最临近cutoff
    #########################################################################
    edge_index = knn(data.pos_j, data.pos, 2)
    i, j = edge_index
    mask = i != j
    i, j = i[mask], j[mask]
    ##########################################################################距离超过cutoff
    max_dist = (data.pos_j[j] - data.pos[i]).norm(dim=-1).max().item()  # 找到最大的距离
    # 按照最大的距离cutoff一次
    if max_dist > cutoff:return False, "距离"
    ##########################################################################完全对称求和为0
    edge_index = radius(data.pos_j, data.pos, r=max(cutoff, max_dist))  # cutoff
    i, j = edge_index
    v_r = (data.pos_j[j] - data.pos[i]) * data.z[j % data.num_nodes].view(-1, 1)  # 矢量方向
    v_sum = torch.zeros((data.z.shape[0], 3),device=device)
    v_sum.index_add_(0, i, v_r)  # 矢量求和
    zero_v = torch.zeros((1, 3),device=device)
    exists = (v_sum == zero_v).all(dim=1).any().item()  # 存在矢量求和为0
    if exists:return False, "对称"
    ##########################################################################只跟自己连接
    for node in range(data.z.shape[0]):
        if check_loop(torch.unique(j[i == node] % data.num_nodes), node):return False, "自环"
    ##########################################################################不受力
    v_r = (data.pos_j[j] - data.pos[i]).abs()  # 矢量绝对值
    abs_v_sum = torch.zeros((data.z.shape[0], 3),device=device)
    abs_v_sum.index_add_(0, i, v_r)  # 矢量绝对值求和
    exists = (abs_v_sum == 0).any().item()  # 矢量绝对值求和结果xyz有一个为0
    if exists:
        ################################################################################不受力但是和F保持一致
        if torch.equal(abs_v_sum == 0, data.force == 0):return False, "单侧 force=0"
        ###############################################################################不受力但是F很小
        data.force[(abs_v_sum == 0) & (data.force < F_threshold)] = 0
        if torch.equal(abs_v_sum == 0, data.force == 0):return False, "force 误差"
        return False, "force 异常"
    ##########################################################################原子半径
    if check_radius:
        i, j = edge_index
        mask = i != j
        i, j = i[mask], j[mask]
    
        z_j = data.z.repeat(data.cell_num.item())
        zr_j = ele_radius[z_j[j] - 1]
        zr_i = ele_radius[data.z[i] - 1]
        r = (data.pos_j[j] - data.pos[i]).norm(dim=-1)
        z_r = (zr_j + zr_i)/2
        radius_z = torch.all(r > z_r).item()
        if not radius_z:return False,"atom radius"
    return True, "正常"

def init_E_emb(data_list):
    count = len(data_list)
    A = np.zeros((count, 105))
    B = np.zeros(count)
    for i,data in tqdm(enumerate(data_list),"E0"):
        B[i] = data.energy.item()
        unique_z, counts = np.unique(data.z, return_counts=True)        
        for j,z in enumerate(unique_z):
            A[i,z-1] =  counts[j]
    E_0 = np.linalg.lstsq(A, B, rcond=None)[0]
    E_0 = np.around(E_0, decimals=4)
    print("init energy E0")
    print(E_0)
    return torch.tensor(E_0)

if __name__ == '__main__':
    pass
