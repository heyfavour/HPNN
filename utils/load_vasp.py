# 读取vasp标准文件
import os, sys
import torch
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import Data
# utils
from utils.load_common import ELEMENT,check_pos,build_pbc,check_healthy
from utils.load_common import init_E_emb

def read_force(raw_dir):
    force_path = os.path.join(raw_dir, "force.arc")
    z_list, force_list, stress_list = [], [], []

    with open(force_path, "r") as f:
        for line in tqdm(f.readlines(), desc="force"):
            line = line.split()
            if len(line) == 3  and line[0] == "Start":
                z,force=[],[]
            elif len(line) == 7:
                xx, xy, xz, yy, yz, zz = [float(i) for i in line[1:7]]
                stress_list.append([[xx, xy, xz],[xy, yy, yz],[xz, yz, zz]])
            elif len(line) == 5:
                force.append((float(line[2]), float(line[3]), float(line[4])))
                z.append(int(line[1]))
            elif len(line) == 3 and line[0] == "End":
                force_list.append(force)
                z_list.append(z)
    return z_list, force_list, stress_list


def read_structure(raw_dir, _z_list, force_list, stress_list, cutoff,device):
    structure_file = os.path.join(raw_dir, "structure.arc")
    idx,data_list,statistics = 0,[],defaultdict(int)
    with tqdm(total=len(force_list),desc="structure") as pbar,open(structure_file, "r") as f:
        for line in f.readlines():
            line = line.split()
            if len(line) == 3 and line[0]=="Start":
                z_list,pos_list,cell = [],[],[]
            elif len(line) == 4 and line[0]=="Energy":
                energy = float(line[2])
            elif len(line) == 4 and line[0] == "lat":
                cell.append((float(line[1]), float(line[2]), float(line[3])))
            elif len(line) == 6 and line[0]=="ele":
                z_list.append(int(line[1]))
                pos_list.append((float(line[2]), float(line[3]), float(line[4])))
            elif len(line) == 3 and line[0] == "End":
                # =============================================================
                assert z_list != len(_z_list[idx]), f"[{idx}]-[{energy}]元素不相等"


                energy = torch.tensor(energy,dtype=torch.float32,device=device)
                z = torch.tensor(z_list, dtype=torch.long,device=device)
                cell = torch.tensor(cell, dtype=torch.float64,device=device)
                pos = torch.tensor(pos_list, dtype=torch.float64,device=device)
                force = torch.tensor(force_list[idx], dtype=torch.float32,device=device)
                stress = torch.tensor(stress_list[idx],dtype=torch.float64,device=device)
                pos = check_pos(pos, cell)  # 校验位置 如果在晶胞外则移动到晶胞内 float64计算
                ##############################################################################
                pos,cell = pos.float(),cell.float()
                image, shift, cell_num, i_num, j_num, pos_j = build_pbc(pos, cell, cutoff=cutoff,device=device)
                #############################################################################frac_pos
                inv_cell = torch.inverse(cell)
                frac_pos = torch.matmul(pos, inv_cell)
                #############################################################################
                #volume = torch.dot(cell[0], torch.cross(cell[1], cell[2]))
                #############################################################################
                cell = cell.unsqueeze(0)
                stress = stress.unsqueeze(0)
                #############################################################################
                data = Data(
                    z=z,
                    pos=pos,
                    energy=energy,
                    force=force,
                    stress=stress,
                    cell=cell,
                    image=image,
                    #shift=shift,
                    frac_pos=frac_pos,#分数坐标
                    cell_num=cell_num,#扩胞次数
                    i_num=i_num,#胞内原子数
                    j_num=j_num,#总原子数
                    pos_j=pos_j,
                    #volume=volume,
                    idx=idx,
                )
                check, info = check_healthy(data, cutoff, device=device)  # 目前不修改数据
                statistics[info]+=1
                data = data.to("cpu")
                if check: data_list.append(data)
                idx += 1
                pbar.update(1)
    assert idx == len(force_list)
    return data_list,statistics


def load(raw_dir, cutoff=5.0,device=torch.device("cpu")):
    z_list, force_list, stress_list = read_force(raw_dir)
    data_list,statistics = read_structure(raw_dir, z_list, force_list, stress_list, cutoff, device)
    print(statistics)
    E_0 = init_E_emb(data_list)
    return data_list,E_0

if __name__ == '__main__':
    FILE_PATH = os.path.dirname(os.path.abspath(__file__))
    load(os.path.join(FILE_PATH, "../data", "train", "raw"), cutoff=5.0)
