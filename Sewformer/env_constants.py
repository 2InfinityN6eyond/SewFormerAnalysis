import os, sys, socket
from glob import glob

# Replacing annoying system.json

hostname = socket.gethostname()
if hostname == "hjpui-MacBookPro.local":
    SEWFORMER_PROJ_ROOT  = "/Users/hjp/HJP/KUAICV/VTO/Sewformer"
    DATASET_ROOT    = "/Users/hjp/HJP/KUAICV/VTO/DATASET/PoC59"
    
elif hostname == "epyc64": # A6000 Ada X 4
    DATASET_ROOT    = "/home/hjp/VTO2025/DATASET/PoC59"
    
elif hostname == "server" : # 4090 X 4
    PYGARMENT_ROOT  = "/media/hjp/db6095ca-a560-4c3a-90ad-b667ec189671/REFERENCES/3D_VTO/GarmentCode/GarmentCode"
    SEWFORMER_PROJ_ROOT  = "/media/hjp/db6095ca-a560-4c3a-90ad-b667ec189671/REFERENCES/SewFormerAnalysis"
    DATASET_ROOT    = "/media/hjp/05aba9a7-0e74-4e54-9bc9-5f11b9c4c757/sewfactory"

elif hostname == "hjp-MS-7D42" : # 3090 X 1
    DATASET_ROOT    = "/media/hjp/efef19d3-9b92-453c-ba04-c205f7233cab/VTO_DATASET/PoC59"

elif hostname == "gpu-1" : # H100 X 8
    PYGARMENT_ROOT  = "/data/hjp/VTO2025/GarmentCodeAnalysis"
    SEWFORMER_PROJ_ROOT  = "/data/hjp/VTO2025/REFERENCES/sewformer"
    DATASET_ROOT    = "/data/hjp/VTO2025/DATASETs/SewFactory"
    
WANDB_USERNAME = ""