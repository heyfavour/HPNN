# 读取lasp标准文件
import os, sys
import torch
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import Data
# utils
from utils.load_common import ELEMENT,Lat,check_pos,build_pbc,check_healthy
from utils.load_common import init_E_emb

def read_force(raw_dir):
    force_path = os.path.join(raw_dir, "force.arc")
    energy_list, force_list, stress_list = [], [], []

    with open(force_path, "r") as f:
        for line in tqdm(f.readlines(), desc="force"):
            line = line.split()
            if line and line[0] == 'For':
                force = []
                energy_list.append(float(line[-1]))
            elif len(line) == 3:force.append((float(line[0]), float(line[1]), float(line[2])))
            elif len(line) == 6:
                xx, xy, xz, yy, yz, zz = [float(i) for i in line]
                stress_list.append([[xx, xy, xz],[xy, yy, yz],[xz, yz, zz]])
            elif len(line) == 0 and force!=[]:
                force_list.append(force)
                force = []
    return energy_list, force_list, stress_list


def read_structure(raw_dir, energy_list, force_list, stress_list, cutoff,device):
    structure_file = os.path.join(raw_dir, "structure.arc")
    idx,data_list,statistics = 0,[],defaultdict(int)
    with open(structure_file, "r") as f:
        for line in tqdm(f.readlines(), desc="structure"):
            line = line.split()
            if len(line) == 4 and line[0] == "Energy":
                energy,structure_id = float(line[-1]),int(line[1])
                z_list,pos_list = [],[]
                end = True
            if len(line) == 10:
                z_list.append(int(ELEMENT[line[0]]))
                pos_list.append((float(line[1]), float(line[2]), float(line[3])))
            elif len(line) == 7:
                cell = Lat(line[1:])
            elif len(line) == 1 and line[0] == "end" and end:
                # =============================================================
                assert energy == energy_list[idx], f"[{idx}]-[{structure_id}]能量不相等"
                assert len(z_list) == len(force_list[idx]), f"[{idx}]-[{structure_id}]数量不相等"


                energy = torch.tensor(energy,dtype=torch.float32,device=device)
                z = torch.tensor(z_list, dtype=torch.long,device=device)
                cell = torch.tensor(cell, dtype=torch.float64,device=device)
                pos = torch.tensor(pos_list, dtype=torch.float64,device=device)
                force = torch.tensor(force_list[idx], dtype=torch.float32,device=device)
                stress = torch.tensor(stress_list[idx],dtype=torch.float32,device=device)
                pos = check_pos(pos, cell)  # 校验位置 如果在晶胞外则移动到晶胞内 float64计算
                ##############################################################################
                pos,cell = pos.float(),cell.float()
                image, shift, cell_num, i_num, j_num, pos_j = build_pbc(pos, cell, cutoff=cutoff,device=device)
                #############################################################################frac_pos
                inv_cell = torch.inverse(cell)
                frac_pos = torch.matmul(pos, inv_cell)
                #############################################################################
                #volume = torch.dot(cell[0], torch.linalg.cross(cell[1], cell[2]))
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
                    structure_id=structure_id,
                )
                check, info = check_healthy(data, cutoff, device)  # 目前不修改数据
                statistics[info]+=1
                data = data.to("cpu")
                if check: data_list.append(data)
                idx += 1
                end = False
    return data_list,statistics


def load(raw_dir, cutoff=5.0,device=torch.device("cpu")):
    energy_list, force_list, stress_list = read_force(raw_dir)
    statistics = defaultdict(int)
    data_list,statistics = read_structure(raw_dir, energy_list, force_list, stress_list, cutoff, device)
    print(statistics)
    E_0 = init_E_emb(data_list)
    return data_list,E_0

if __name__ == '__main__':
    FILE_PATH = os.path.dirname(os.path.abspath(__file__))
    load(os.path.join(FILE_PATH, "../data", "train", "raw"), cutoff=5.0,deivce=torch.device("cpu"))
