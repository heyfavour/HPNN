import logging
import os
import sys



lab_info = """\
+-------------------------------------------------------------------------------------------+
|                    Welcome to Fantastic Atomic World in Neural Network                    |
|                                          -------------                                    |
|                                             L A S P                                       |
|                                          -------------                                    |
|               Large-scale Atomic Simulation with neural network Potential                 |
|                      Version 4.0.0  (   GPU + CPU       )    Dec. 2024                    |
|                                       Shanghai, China                                     |
|                               More on Website: www.lasphub.com                            |
|                                                                                           |
|     Please cite: S.D. Huang, C. Shang, P.L. Kang, X.J. Zhang and Z.P. Liu ,               |
|                  "LASP: Fast Global Potential Energy Surface Exploration" ,               |
|                  WIREs. Comp. Mole. Sci. 2019, 9, e1415 (DOI:10.1002/wcms.1415)           |
|                                                                                           |
|                  Z.X. Yang, Z.X. Wang, X.T. Xie, C. Shang, Z.P. Liu, 2024, submitted      |
+-------------------------------------------------------------------------------------------+
"""



class GPUFilter(logging.Filter):
    """自定义过滤器，仅允许 gpu_id=0 的日志通过"""
    def __init__(self, dist_id,local_id):
        super().__init__()
        self.dist_id = dist_id
        self.local_id = local_id

    def filter(self, record):
        return self.dist_id == 0

class GPUHandler(logging.FileHandler):
    """负责日志处理的 Handler 类"""
    def __init__(self, dist_id,local_id, log_dir='./logs'):
        # 根据 GPU ID 创建单独的日志文件
        filename = os.path.join(log_dir, f"gpu_{dist_id}.debug")
        super().__init__(filename=filename,mode="w")  # 初始化 FileHandler

        # 设置格式
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        self.setFormatter(formatter)

class GpuLogger(logging.Logger):
    """负责日志记录的 Logger 类，继承自 logging.Logger"""
    def __init__(self, dist_id=0, local_id=0, log_level=logging.DEBUG, log_dir='./logs/debug'):
        super().__init__(name=f"gpu_{dist_id}", level=log_level)
        self.dist_id = dist_id
        self.local_id = local_id

        # 仅在初次配置时添加 handler 和 filter
        if not self.hasHandlers():
            os.makedirs(log_dir, exist_ok=True)

            # 创建并添加控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)  # 控制台处理器
            console_handler.setLevel(logging.INFO)  # 仅记录 INFO 级别的日志
            console_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_formatter)

            # 创建并添加文件处理器
            debug_handler = GPUHandler(dist_id,local_id, log_dir)  # 文件处理器
            debug_handler.setLevel(logging.DEBUG)  # 记录 DEBUG 级别的日志

            self.addHandler(console_handler)
            self.addHandler(debug_handler)

            # 添加过滤器
            gpu_filter = GPUFilter(dist_id,local_id)
            console_handler.addFilter(gpu_filter)  # 控制台处理器应用过滤器
            #debug_handler.addFilter(gpu_filter)  # 文件处理器也应用过滤器
        self.info(lab_info)

if __name__ == "__main__":
    # 测试代码
    gpu_id = 0  # 你可以在这里更改为 1 来测试不同的行为
    logger = GpuLogger(gpu_id)

    logger.info("This is an info message1.")
    logger.debug("This is a debug message2.")

