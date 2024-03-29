#from .adafactor import Adafactor as GaLoreAdafactor
import os
gpu_ids = "0,1,2"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
print(f"CUDA set to gpu_ids: {gpu_ids}")
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
from .adamw import AdamW as GaLoreAdamW
# from .adamw8bit import AdamW8bit as GaLoreAdamW8bit