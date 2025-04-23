import os
import torch
import importlib

from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
# utils 
from utils.common import read_config
# argparse
import argparse


import warnings

# 指定要忽略的特定警告消息
warnings.filterwarnings("ignore", message="Length of split at index 1 is 0. This might result in an empty dataset.")


class PBCDataset(InMemoryDataset):
    def __init__(self, root: str, cutoff=5.0, device=torch.device("cpu"),transform=None, pre_transform=None, pre_filter=None):
        self.cutoff = cutoff
        self.device = device
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)
        self.E_0 = torch.load(self.processed_paths[1],weights_only=False)

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
        return ['data.pt','E_0.pt']

    def process(self):
        config = read_config()
        module = importlib.import_module(f"utils.load_{config['file']}")
        _list,E_0 = module.load(self.raw_dir, cutoff=self.cutoff,device=self.device)


        torch.save(E_0, self.processed_paths[1])
        torch.save(self.collate(_list), self.processed_paths[0])


def rank_generator(rank=0,seed=99):
    generator = torch.Generator()
    generator.manual_seed(seed + rank)
    return generator


def dataloader(batch_size,percent,cutoff=5.0,ddp=False,rank=0,data_dir="train"):
    local_path= os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(local_path, "data",data_dir)
    if ddp and  (not os.path.exists(os.path.join(path,'processed','data.pt'))):raise Exception("未执行python load_data.py")
    dataset = PBCDataset(path,cutoff)
    train_dataset, valid_dataset = random_split(dataset, percent)
    if ddp: 
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        generator = rank_generator(rank=rank)
        train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=True,sampler=train_sampler,generator=generator)

        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=1,pin_memory=True,sampler=valid_sampler)
    else:
        train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=True)
        valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=1,pin_memory=True)
    info = {
        "train_count": len(train_dataset),
        "valid_count": len(valid_dataset),
        "E_0":dataset.E_0,
    }
    return train_loader, valid_loader, info


def load_dataset(cutoff=5.0,device=torch.device("cpu")):
    local_path= os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(local_path, "data","train")
    dataset = PBCDataset(path,cutoff,device=device)
    info = {
        "dataset": len(dataset),
        "E_0":dataset.E_0,
    }
    print(info)
    return dataset



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device.isdigit() else "cpu")
    print(f"device: {device}")
    config = read_config()
    cutoff = config['cutoff']
    print(f"cutoff: {cutoff}")
    load_dataset(cutoff=cutoff,device=device)
