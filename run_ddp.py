import os
import datetime
import math
import torch
import warnings
import torch.optim as optim
# amp
from torch import amp
# train
from model import PTSDGraphNet
from load_data import dataloader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# utils
from utils.run_common import set_seed,init_epoch,collect_batch,gpu_loss,save_model,check_best,interval_save
from utils.common import read_config
from utils.logger import GpuLogger



########################################################################################
# 多机多卡 loss
# 所有gpu的rmse取平均
#######################################################################################
#忽略warning
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

def gather_loss(loss,local_rank,gpu_nums):
    loss = torch.tensor(loss).to(local_rank)
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    loss = loss / gpu_nums
    return loss.cpu().item()

def gather_ddp_loss(loss,local_rank,gpu_nums):
    for k,v in loss.items():
        loss[k] = gather_loss(v,local_rank, gpu_nums)
    return loss


if __name__ == '__main__':
    ####################################################################################################### DDP
    dist.init_process_group(backend='nccl',timeout=datetime.timedelta(seconds=1800))  # nccl是GPU设备上最快、最推荐的后端
    # GPU_ID
    dist_rank,local_rank = int(dist.get_rank()),int(os.environ["LOCAL_RANK"])
    log = GpuLogger(dist_id=dist_rank,local_id=local_rank)
    log.info(f"[dist_rank] {dist_rank}   [local_rank] {local_rank}")
    torch.cuda.set_device(local_rank)
    run_time = datetime.datetime.now()
    log.info(f"run_time: {run_time}")
    seed_id  = 99
    set_seed(seed_id) #随机种子
    log.info(f"seed_id: {seed_id}")
    ####################################################################################################### 参数
    config_data = read_config()
    #######################################################################################################
    batch_size = config_data["batch_size"]
    cutoff = config_data["cutoff"]
    log.info(f"batch_size: {batch_size}")
    log.info(f"cutoff: {cutoff}")
    start_time = datetime.datetime.now()
    percent = [config_data["train_percent"],config_data["valid_percent"]]
    train_loader, valid_loader, info = dataloader(batch_size,percent,cutoff=cutoff,ddp=True,rank=dist_rank)
    end_time = datetime.datetime.now()
    log.info(f"load time: {end_time - start_time}")
    log.info(f"info [train]:{info['train_count']} [valid]:{info['valid_count']}")
    ####################################################################################################### 模型参数
    node_dim = config_data["node_dim"]
    num_layers = config_data["num_layers"]
    pair_dim = config_data["pair_dim"]
    num_rbf = config_data["num_rbf"]
    sph = config_data['sph']
    dropout = config_data["dropout"]
    E_factor = config_data["E_factor"]
    F_factor = config_data["F_factor"]
    S_factor = config_data["S_factor"]

    log.info(f"node_dim: {node_dim}")
    log.info(f"num_layers: {num_layers}")
    log.info(f"pair_dim: {pair_dim}")
    log.info(f"num_rbf: {num_rbf}")
    log.info(f"sph: {sph}")
    log.info(f"dropout: {dropout}")
    log.info(f"E_factor: {E_factor}")
    log.info(f"F_factor: {F_factor}")
    log.info(f"S_factor: {S_factor}")
    ####################################################################################################### init model
    model = PTSDGraphNet(cutoff=cutoff, node_dim=node_dim, num_layers=num_layers, pair_dim=pair_dim, num_rbf=num_rbf,sph=sph,dropout=dropout,E_0=info["E_0"])
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    ####################################################################################################### lr
    lr = config_data["learning_rate"]
    gpu_nums = dist.get_world_size()
    log.info(f"lr: {lr}")
    log.info(f"gpu_nums: {gpu_nums}")
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    batch_count = math.ceil(info["train_count"] / (batch_size * gpu_nums))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=batch_count, T_mult=2)
    interval = int(2560/batch_size)
    #######################################################################################################
    log.info(f"参数总量: {sum(p.numel() for p in model.parameters())}")
    criterion = torch.nn.SmoothL1Loss().to(local_rank)
    log.info(f"criterion: {criterion}")
    train_path = f'./model_pth/{node_dim}_{num_layers}_{pair_dim}_train_{E_factor}E{F_factor}F{S_factor}S.pth'
    valid_path = f'./model_pth/{node_dim}_{num_layers}_{pair_dim}_valid_{E_factor}E{F_factor}F{S_factor}S.pth'
    log.info(f"train_path: {train_path}")
    log.info(f"valid_path: {valid_path}")

    max_clip = config_data["max_clip"]
    epoch_num = config_data["epoch_num"]
    save_interval = config_data["save_interval"]
    log.info(f"max_clip: {max_clip}")
    log.info(f"epoch_num: {epoch_num}")
    log.info(f"save_interval: {save_interval}")

    best_train,best_valid = float("inf"),float("inf")
    scaler = amp.GradScaler()

    start_epoch = 0


    if config_data["retrain"]:
        retrain_model = train_path
        start_epoch = config_data["retrain_epoch"]
        scheduler.step(start_epoch*batch_count)
        log.info(f"retrain model {retrain_model}")
        if not os.path.exists(retrain_model):raise Exception("retrain需要加载得模型不存在") 
        model.module.load_state_dict(torch.load(train_path,weights_only=True,map_location=f"cuda:{local_rank}"))
        dist.barrier()



    for epoch in range(start_epoch,epoch_num):
        ########################################################################################################## train
        start_time = datetime.datetime.now()
        train_loader.sampler.set_epoch(epoch)
        _energy,_force,_stress,_node_num = init_epoch(local_rank)
        model.train()
        mark = "TRAIN"
        log.info("-"*128)
        for idx, data in enumerate(train_loader):
            data = data.to(local_rank)
            optimizer.zero_grad()
            # 混合精度训练
            with amp.autocast('cuda'):
                energy, force, stress = model(data)
                energy_loss = criterion(energy, data.energy.view(-1, 1))
                force_loss = criterion(force, data.force)
                stress_loss = criterion(stress, data.stress)
                loss = energy_loss * E_factor + force_loss * F_factor + stress_loss * S_factor
            scaler.scale(loss).backward()
            # clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip)
            # amp update
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            _energy,_force,_stress,_node_num = collect_batch(_energy,_force,_stress,_node_num,data,(energy,force,stress))

            if idx % interval == 0:
                log.debug(f"[GPU] {dist_rank} [IDX]:{idx:0>4} [loss]:{loss.item():.6f} [energy]:{energy_loss.item():.5f} [force]:{force_loss:.5f} [stress]:{stress_loss:.6f} [lr]:{optimizer.param_groups[0]['lr']:.6f}")
        end_time = datetime.datetime.now()
        time = (end_time,start_time)
        loss = gpu_loss(epoch,_energy, _force,_stress, _node_num,mark,log)
        loss = gather_ddp_loss(loss,local_rank,gpu_nums)#ddploss
        best_train = save_model(epoch,loss,best_train,model,train_path,(end_time-start_time),mark,log,ddp=True)
        interval_path = f'./model_pth/debug/{node_dim}_{num_layers}_{pair_dim}_train_{E_factor}E{F_factor}F{S_factor}S_{epoch}.pth'
        interval_save(epoch,model,interval_path,save_interval,log,ddp=True)
        dist.barrier()
        if info["valid_count"] == 0:continue
        ########################################################################################################## valid
        #存在bug,目前验证集只master执行不能超过32 batch
        start_time = datetime.datetime.now()
        valid_loader.sampler.set_epoch(epoch)
        _energy,_force,_stress,_node_num = init_epoch(local_rank)
        model.eval()
        mark = "VALID"
        for idx, data in enumerate(valid_loader):
            optimizer.zero_grad()
            data.to(local_rank)
            with amp.autocast('cuda'):
                energy, force, stress = model(data)  # [bs,1]
            _energy,_force,_stress,_node_num = collect_batch(_energy,_force,_stress,_node_num,data,(energy,force,stress))
        end_time = datetime.datetime.now()
        loss = gpu_loss(epoch,_energy, _force,_stress, _node_num,mark,log)
        loss = gather_ddp_loss(loss,local_rank,gpu_nums)#ddploss
        best_valid = save_model(epoch,loss,best_valid,model,valid_path,(end_time-start_time),mark,log,ddp=True)
        dist.barrier()
    log.info(f"本次训练耗时: {str(datetime.datetime.now() - run_time)[:-4]}")
    set_seed(seed_id)
    if dist_rank == 0:check_best(cutoff,node_dim,num_layers,pair_dim,num_rbf,sph,train_path,local_rank,batch_size,percent,log)
