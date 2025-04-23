### 训练 
```
1.准备数据 ./data/train/raw force.arc structure.arc
2.处理数据 python load_data.py
  # python load_data.py --device cpu #使用cpu 默认是cpu
  # python load_data.py --device 0   #使用0号显卡
3.配置 config.yml
4.python gen_bash.py # 生成队列脚本
5.sh run.sh
```
### 导出model.pth->laspgpu
```
1.python -m  utils.jit_save
  # python -m utils.jit_save --name xxx.pth #指定导出pth文件名,默认是lasp_model.pth
2.查看./model_pth/下的推理的model,用于laspgpu
```

### 目录结构
```
src:[./src]
|--__init__.py
|--data
|      |--train                                         # 训练目录
|      |      |--raw
|      |      |      |--force.arc                       # 力文件
|      |      |      |--structure.arc                   # 结构文件
|--bash                                                 # 存放提交队列的bash脚本
|--config.yml                                           # 配置文件
|--gen_bash.py                                          # 根据congfig.yml 生成所有执行脚本
|--run.sh                                               # sh run.sh 任务提交
|--run_amp.py                                           # 单卡 main
|--run_ddp.py                                           # 多卡 main
|--model.py                                             # 模型
|--model_block.py                                       # 模型
|--model_pth
|      |--256_3_128_train_1E5F1S.pth                    # train best model
|      |--256_3_128_valid_1E5F1S.pth                    # valid best model
|--logs
|      |--CuCHON_512_3_128_20240301.log                 # log
|      |--debug                                         # gpu日志明细,用于回溯
|--utils
|      |--__init__.py
|      |--bash.py                                       # bash模板
|      |--common.py
|      |--run_common.py
|      |--load_common.py
|      |--load_valid.py                                 
|      |--load_arc.py
|      |--load_vasp.py
|      |--jit_code                                      # jit编译的model code
|      |      |--__init__.py
|      |      |--model_block.py
|      |      |--model.py
|      |--valid.py                                      # 验证已有模型对某个数据集的推理效果
|      |--jit_save.py                                   # 生成lasp-gpu可以推理的model.pth 需要在里面修改参数
|--load_data.py                                         # load pyg格式的数据,会对data/train/raw下的arc文件进行一定的预处理
|--readme.md
```

### 环境安装
#### miniconda3
```
mkdir -p ./miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda3/miniconda.sh
bash ./miniconda3/miniconda.sh -b -u -p ./miniconda3
rm -rf ./miniconda3/miniconda.sh

./miniconda3/bin/conda init bash
```

#### python
```
conda create --name lasp python=3.11
# 激活conda lasp
conda activate lasp
```

```
# cuda驱动版本12.4
# pytorch 需要注意pyg支持的最高torch版本 目前只支持到2.4.*
# pytorch 从2.3以后开始支持numpy2.0
# 当前最新torch 2.4.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# pyg
pip install torch_geometric
# 注意不要安装pyg_lib
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
# e3nn 用于球谐
pip install e3nn
# yml 用于读取配置 安装torch2.4会自动安装pyyaml 安装低版本torch需要自己安装pyyaml
# pip install pyyaml
```
