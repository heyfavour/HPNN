"""
train 重新train的RMSE MAE
valid 验证数据的RMSE MAE
"""
import os,sys
import math
import torch
import time
import datetime
import torch.optim as optim

# model
from model import PTSDGraphNet
# utils
from load_data import dataloader,PBCDataset
from torch_geometric.loader import DataLoader
from utils.run_common import set_seed,deal_loader,gpu_loss
from torch.utils.data import random_split, Subset
from utils.common import read_config
# predict
from tqdm import tqdm
from torch_geometric.data import Data
from utils.load_common import ELEMENT,Lat,check_pos,build_pbc
from torch_geometric.data import InMemoryDataset
# analysis
from collections import defaultdict
# argparse
import argparse



def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行被装饰的函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时
        print(f"load time：{elapsed_time:.4f} 秒")
        return result
    return wrapper

# =======================================================================================================DATALOADER
# train
@timer
def train_dataloader(batch_size,percent,cutoff=5.0,data_dir=None):
    local_path= os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(local_path,"..", "data",data_dir)
    dataset = PBCDataset(path,cutoff)
    train_dataset, valid_dataset = random_split(dataset, percent)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False,num_workers=1,pin_memory=True)
    valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=1,pin_memory=True)
    info = {
        "train_count": len(train_dataset),
        "valid_count": len(valid_dataset),
    }
    print(info)
    return train_loader, valid_loader, info

# valid 
@timer
def valid_dataloader(batch_size,cutoff=5.0,data_dir=None):
    local_path= os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(local_path,"..", "data",data_dir)
    dataset = PBCDataset(path,cutoff)
    valid_loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=1,pin_memory=True)
    info = {
        "valid_count": len(dataset),
    }
    print(info)
    return valid_loader, info

# predict
@timer
def read_arc(raw_dir,cutoff):
    device = torch.device("cpu")
    structure_file = os.path.join(raw_dir, "structure.arc")
    idx,data_list = 0,[]
    with open(structure_file, "r") as f:
        for line in tqdm(f.readlines(), desc="structure"):
            line = line.split()
            if len(line) in  {4,5} and line[0] == "Energy":
                energy, structure_id = float(line[-1]), int(line[1])
                z_list,pos_list = [],[]
                end = True
            if len(line) == 10:
                z_list.append(int(ELEMENT[line[0]]))
                pos_list.append((float(line[1]), float(line[2]), float(line[3])))
            elif len(line) == 7:
                cell = Lat(line[1:])
            elif len(line) == 1 and line[0] == "end" and end:
                z = torch.tensor(z_list, dtype=torch.long,device=device)
                cell = torch.tensor(cell, dtype=torch.float64,device=device)
                pos = torch.tensor(pos_list, dtype=torch.float64,device=device)
                pos = check_pos(pos, cell)  # 校验位置 如果在晶胞外则移动到晶胞内 float64计算
                ##############################################################################
                pos,cell = pos.float(),cell.float()
                image, shift, cell_num, i_num, j_num, pos_j = build_pbc(pos, cell, cutoff=cutoff,device=device)
                #############################################################################frac_pos
                inv_cell = torch.inverse(cell)
                frac_pos = torch.matmul(pos, inv_cell)
                #############################################################################
                volume = torch.dot(cell[0], torch.cross(cell[1], cell[2]))
                #############################################################################
                cell = cell.unsqueeze(0)
                #############################################################################
                data = Data(
                    z=z,
                    pos=pos,
                    cell=cell,
                    image=image,
                    shift=shift,
                    frac_pos=frac_pos,#分数坐标
                    cell_num=cell_num,#扩胞次数
                    i_num=i_num,#胞内原子数
                    j_num=j_num,#总原子数
                    pos_j=pos_j,
                    volume=volume,
                    idx=idx,
                    structure_id=structure_id,
                )
                data_list.append(data)
                idx += 1
                end = False
    return data_list


class PredictDataset(InMemoryDataset):
    def __init__(self, root: str, cutoff=5.0, transform=None, pre_transform=None, pre_filter=None):
        self.cutoff = cutoff
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed")

    @property
    def raw_file_names(self):
        names = ['structure', 'force']
        return [f'{self.name}.arc' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        config = read_config()
        if config['file'] == "arc":_list = read_arc(self.raw_dir,cutoff=self.cutoff)
        torch.save(self.collate(_list), self.processed_paths[0])
 

def predict_dataloader(batch_size,cutoff=5.0,data_dir=None):
    local_path= os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(local_path,"..", "data",data_dir)
    dataset = PredictDataset(path,cutoff)
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=1,pin_memory=True)
    info = {
        "count": len(dataset),
    }
    print("info:",info)
    return data_loader, info

# =======================================================================================================INFERENCE
def predict():
    torch.set_printoptions(precision=4,sci_mode=False)  
    model.eval()
    for idx, data in enumerate(predict_loader):
        data = data.to(device)
        energy, force, stress = model(data)
        print(f"===================={idx}==========================")
        print(energy)
        print(force)
        print(stress)



def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode',choices=['train', 'valid', 'predict','analysis'],default='train')
    parser.add_argument('--data_dir',default='train')
    parser.add_argument('--cuda',type=int,default=0)
    args = parser.parse_args()
    return args

def init_model(config,device):
    ####################################################################################################### 模型参数
    node_dim = config["node_dim"]
    num_layers = config["num_layers"]
    pair_dim = config["pair_dim"]
    num_rbf = config["num_rbf"]
    sph = config['sph']
    E_factor = config["E_factor"]
    F_factor = config["F_factor"]
    S_factor = config["S_factor"]
    dropout = 0

    print("node_dim:",node_dim)
    print("num_layers:",num_layers)
    print("pair_dim:",pair_dim)
    print("num_rbf:",num_rbf)
    print("sph:",sph)
    print("dropout:",dropout)
    print("E_factor:",E_factor)
    print("F_factor:",F_factor)
    print("S_factor:",S_factor)
    train_path = f'./model_pth/{node_dim}_{num_layers}_{pair_dim}_train_{E_factor}E{F_factor}F{S_factor}S.pth'
    valid_path = f'./model_pth/{node_dim}_{num_layers}_{pair_dim}_valid_{E_factor}E{F_factor}F{S_factor}S.pth'
    load_path = train_path
    print("load_path",load_path)
    ####################################################################################################### init model
    model = PTSDGraphNet(cutoff=cutoff, node_dim=node_dim, num_layers=num_layers, pair_dim=pair_dim, num_rbf=num_rbf,sph=sph,dropout=dropout)
    model.load_state_dict(torch.load(load_path,map_location=device,weights_only=False))
    model = model.to(device)
    print("参数总量:",sum(p.numel() for p in model.parameters()))
    return model

def print_loss(loss,mark):
    rmse_mole,rmse_atom,rmse_force,rmse_stress = loss['rmse_mole'],loss['rmse_atom'],loss['rmse_force'],loss['rmse_stress']
    mae_mole,mae_atom,mae_force,mae_stress = loss['mae_mole'],loss['mae_atom'],loss['mae_force'],loss['mae_stress']
    
    rmse_info = f"[{mark}] [RMSE] [energy] [mol] {rmse_mole:.6f} [atom] {rmse_atom:.6f} [force] {rmse_force:.5f} [stress] {rmse_stress:.4f}"
    mae_info = f"[{mark}] [MAE ] [energy] [mol] {mae_mole:.6f} [atom] {mae_atom:.6f} [force] {mae_force:.5f} [stress] {mae_stress:.4f}"
    print(rmse_info)
    print(mae_info)

def dataloader_group(batch_size,cutoff,data_dir="train"):
    local_path= os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(local_path,"..", "data",data_dir)
    start_time = datetime.datetime.now()
    dataset = PredictDataset(path,cutoff)
    end_time = datetime.datetime.now()
    print(f"load time {data_dir}",end_time-start_time)
    group_subset_idx = defaultdict(list)
    ele = {v:k for k,v in ELEMENT.items()}
    for idx,data in enumerate(dataset):
        name = "".join([ele[i] for i in  torch.unique(data.z).sort()[0].tolist()])
        group_subset_idx[name].append(idx)
    return dataset,group_subset_idx

    
if __name__ == '__main__':
    args = parse_config()
    print("pid:", os.getpid())
    print("run time:",datetime.datetime.now())
    set_seed(99) #随机种子
    config = read_config()
    batch_size = config['batch_size'] 
    cutoff = config["cutoff"]
    print("batch_size:",batch_size)
    print("cutoff:",cutoff)
    ####################################################################################################### 参数
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    model = init_model(config,device)
    set_seed(99) #随机种子
    if args.mode == "train":
        percent = [config["train_percent"],config["valid_percent"]]
        train_loader, valid_loader, info = train_dataloader(batch_size,percent,cutoff=cutoff,data_dir=args.data_dir)#复现数据
        train_loss = deal_loader(model,train_loader,device)
        print_loss(train_loss,'TRAIN')
        if info['valid_count'] > 0:
            valid_loss = deal_loader(model,valid_loader,device)
            print_loss(valid_loss,'VALID')
    if args.mode == "valid":
        valid_loader, info = valid_dataloader(batch_size,cutoff=cutoff,data_dir=args.data_dir)
        valid_loss = deal_loader(model,valid_loader,device)
        print_loss(valid_loss,'VALID')
    if args.mode == "predict":
        predict_loader, info = predict_dataloader(1,cutoff=cutoff,data_dir=args.data_dir)
        predict()
    if args.mode == "analysis":
        dataset,group_subset_idx = dataloader_group(batch_size,cutoff=cutoff,data_dir=args.data_dir)
        for name,indices in group_subset_idx.items():
            subset = Subset(dataset, indices)
            count = len(indices)
            data_loader = DataLoader(subset, batch_size=batch_size)
            print("-"*87)
            print(f"[   subset   ] {name:<20} [count] {count:<10}")
            loss = deal_loader(model,data_loader,device)
            print_loss(loss,f"{name:<12}")
