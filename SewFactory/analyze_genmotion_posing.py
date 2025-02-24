import os, sys

genmotion_path = r"C:\Users\dytpq0916\VTO2025\TOOLs\GenMotion"
sys.path.append(genmotion_path)

print(sys.path)


import maya.cmds as cmds

import numpy as np
from tqdm.auto import tqdm

# import GenMotion modules

from genmotion.dataset.amass_params import SMPL_H_SKELETON # recognize the skeleton type as SMPL_H
from genmotion.render.maya.utils import MayaController


