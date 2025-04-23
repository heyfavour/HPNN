import os
import random
import torch
import numpy as np
import logging

from model import PTSDGraphNet
from load_data import dataloader 

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



@torch.no_grad()
def init_epoch(device):
    energy = [torch.Tensor([]).to(device), torch.Tensor([]).to(device)]
    force = [torch.Tensor([]).to(device), torch.Tensor([]).to(device)]
    stress = [torch.Tensor([]).to(device), torch.Tensor([]).to(device)]
    node_num = torch.Tensor([]).to(device)
    return energy,force,stress,node_num


@torch.no_grad()
def collect_batch(energy,force,stress,node_num,data,batch):
    energy[0] = torch.cat([energy[0], batch[0].float().detach_().view(-1, 1)], dim=0)
    energy[1] = torch.cat([energy[1], data.energy.view(-1, 1)], dim=0)
    force[0] = torch.cat([force[0], batch[1].float().detach_().view(-1, 3)], dim=0)
    force[1] = torch.cat([force[1], data.force.view(-1, 3)], dim=0)
    stress[0] = torch.cat([stress[0], batch[2].float().detach_().view(-1,3,3)], dim=0)
    stress[1] = torch.cat([stress[1], data.stress.view(-1,3,3)], dim=0)
    node_num = torch.cat([node_num, data.i_num.view(-1, 1)], dim=0)
    return energy,force,stress,node_num

@torch.no_grad()
def gpu_loss(epoch,energy,force,stress,node_num,mark="",log=None):
    # RMSE
    rmse_mole = torch.sqrt(torch.mean((energy[0] - energy[1]) ** 2)).cpu().item()
    rmse_atom = torch.sqrt(torch.sum((energy[0] - energy[1]) ** 2 / node_num.view(-1, 1)) / torch.sum(node_num)).cpu().item()
    rmse_force = torch.sqrt(torch.mean((force[0] - force[1]) ** 2)).cpu().item()
    rmse_stress = torch.sqrt(torch.mean(((stress[0] - stress[1]) * 160.21766208) ** 2)).cpu().item()
    if log:log.debug(f"[{mark}] [GPU  ] {epoch} [RMSE] [energy] [mol] {rmse_mole:.6f} [atom] {rmse_atom:.6f} [force] {rmse_force:.5f} [stress] {rmse_stress:.4f}")
    # MAE
    mae_mole = torch.mean(torch.abs(energy[0] - energy[1])).cpu().item()
    mae_atom = torch.mean(torch.abs(energy[0] - energy[1])/node_num.view(-1, 1)).cpu().item()
    mae_force = torch.mean(torch.abs(force[0] - force[1])).cpu().item()
    mae_stress = torch.mean(torch.abs((stress[0] - stress[1]) * 160.21766208)).cpu().item()
    if log:log.debug(f"[{mark}] [GPU  ] {epoch} [MAE ] [energy] [mol] {mae_mole:.6f} [atom] {mae_atom:.6f} [force] {mae_force:.5f} [stress] {mae_stress:.4f}")
    loss = {
        "rmse_mole" : rmse_mole,
        "rmse_atom" : rmse_atom,
        "rmse_force" : rmse_force,
        "rmse_stress" : rmse_stress,
        "mae_mole" : mae_mole,
        "mae_atom" : mae_atom,
        "mae_force" : mae_force,
        "mae_stress" : mae_stress,
    }
    return loss



@torch.no_grad()
def save_model(epoch,loss,best_loss,model,save_path,time,mark,log,ddp=False):
    rmse_mole,rmse_atom,rmse_force,rmse_stress = loss['rmse_mole'],loss['rmse_atom'],loss['rmse_force'],loss['rmse_stress']
    mae_mole,mae_atom,mae_force,mae_stress = loss['mae_mole'],loss['mae_atom'],loss['mae_force'],loss['mae_stress']
    
    rmse_info = f"[{mark}] [EPOCH] {epoch} [RMSE] [energy] [mol] {rmse_mole:.6f} [atom] {rmse_atom:.6f} [force] {rmse_force:.5f} [stress] {rmse_stress:.4f}  [time] {str(time)[:-4]}"
    mae_info = f"[{mark}] [EPOCH] {epoch} [MAE ] [energy] [mol] {mae_mole:.6f} [atom] {mae_atom:.6f} [force] {mae_force:.5f} [stress] {mae_stress:.4f}"

    loss = rmse_atom*10 + rmse_force
    save_info = ""
    if log.dist_id == 0 and loss < best_loss:
        best_loss = loss
        if ddp:torch.save(model.module.state_dict(), save_path)
        else:torch.save(model.state_dict(), save_path)
        save_info = " [BEST SAVE]"
    log.info(rmse_info  + save_info)
    log.info(mae_info)
    return best_loss

def interval_save(epoch,model,save_path,interval,log,ddp=False):
    if interval == -1:return
    if (log.dist_id == 0) and ((epoch+1) % interval == 0) and (epoch > 0):
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        if ddp:torch.save(model.module.state_dict(), save_path)
        else:torch.save(model.state_dict(), save_path)

    
def deal_loader(model,data_loader,device):
    model.eval()
    #start_time = datetime.datetime.now()
    _energy,_force,_stress,_node_num = init_epoch(device)
    for idx, data in enumerate(data_loader):
        data = data.to(device)
        energy, force, stress = model(data)
        _energy,_force,_stress,_node_num = collect_batch(_energy,_force,_stress,_node_num,data,(energy,force,stress))
    loss = gpu_loss(0,_energy, _force,_stress, _node_num)
    #end_time = datetime.datetime.now()
    return loss


def load_model_with_path(cutoff,node_dim,num_layers,pair_dim,num_rbf,sph,load_path,device):
    model = PTSDGraphNet(cutoff=cutoff, node_dim=node_dim, num_layers=num_layers, pair_dim=pair_dim, num_rbf=num_rbf,sph=sph,dropout=0)
    if isinstance(device, int):device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(load_path,map_location=device,weights_only=False))
    model = model.to(device)
    return model

# train ======================================================================================================

def check_best(cutoff,node_dim,num_layers,pair_dim,num_rbf,sph,load_path,device,batch_size,percent,log,data_dir="train"):
    train_loader, valid_loader, info = dataloader(batch_size,percent,cutoff=cutoff,ddp=False)
    model = load_model_with_path(cutoff,node_dim,num_layers,pair_dim,num_rbf,sph,load_path,device)
    loss_train = deal_loader(model,train_loader,device)
    if info['valid_count']>0:loss_valid = deal_loader(model,valid_loader,device)
    log.info("")
    # 9, 17, 19, 15, 19,
    log.info("+---------+-----------------+-------------------+---------------+-------------------+")
    log.info("|         |  structure num  |  RMSE E/meV/atom  |  RMSE F/eV/A  |  RMSE Stress GPA  |")
    log.info("+---------+-----------------+-------------------+---------------+-------------------+")
    log.info(f"|  TRAIN  |  {info['train_count']:>11}    |  {loss_train['rmse_atom']*1000:>13.4f}    |  {loss_train['rmse_force']:>9.4f}    |  {loss_train['rmse_stress']:>13.4f}    |")
    if info['valid_count']>0:
        log.info(f"|  VALID  |  {info['valid_count']:>11}    |  {loss_valid['rmse_atom']*1000:>13.4f}    |  {loss_valid['rmse_force']:>9.4f}    |  {loss_valid['rmse_stress']:>13.4f}    |")
    log.info("+---------+-----------------+-------------------+---------------+-------------------+")

    

