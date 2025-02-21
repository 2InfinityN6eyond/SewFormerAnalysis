# Basic
# from __future__ import print_function
# from __future__ import division

import os, sys
from pathlib import Path

from imp import reload

package_path = (Path(__file__).parent.parent / "packages").resolve()
mayaqltools_path = package_path / "mayaqltools"

sys.path.append(str(package_path))
sys.path.append(str(mayaqltools_path))


from random import random, uniform
from re import L
from unittest import skipUnless
import numpy as np
import random
import json

from dataclasses import dataclass
from datetime import datetime
from glob import glob
import time
# import argparse
from copy import deepcopy


# Maya
from maya import cmds, mel
from maya import OpenMaya
import maya.api.OpenMaya as OM
import pymel.core as pm
pm.loadPlugin("fbxmaya") # LOAD PLUGIN

# Arnold
# import mtoa.utils as mutils
# from mtoa.cmds.arnoldRender import arnoldRender
# import mtoa.core


# from mayaqltools import utils
# reload(utils)



from mayaqltools import utils
reload(utils)

from mayaqltools import fbx_animation

print("RUNNING ANYWAY")

if __name__ == "__main__":
    @dataclass
    class Args:
        base_config     : str = "meta_infos/configs/anime_config.json"
        base_fbx        : str = "meta_infos/fbx_metas/basicModel_f_lbs_10_207_0_v1.0.2.fbx"  #  "meta_infos/fbx_metas/SMPLH_female_010_207.fbx"
        skin_textures   : str = "examples/skin_textures"
        pose_root       : str = "examples/human_poses"
        output          : str = "test/posed_fbxs"
        animated        : str = ""

    args = Args()

    print(args)


    config = {
        "joints_mat_path": "meta_infos/fbx_metas/joints_mat_SMPLH.npz",
        "body_file": "meta_infos/fbx_metas/basicModel_f_lbs_10_207_0_v1.0.2.fbx",
        "texture_file": "examples/skin_textures",
        "pose_file": "",
        "animated": False,
        "num_frames": 4,
        "export_log": False,
    }


    cmds.file(
        config["body_file"],
        i=True, type='Fbx',
        ignoreVersion=True, groupReference=True, groupName="bodyfbx"
    )

    bodyfbx = [bf for bf in cmds.ls(transforms=True) if "bodyfbx" in bf][0]

    print(bodyfbx)
    print("="*100)
    
    bodyfbx = cmds.rename(bodyfbx, 'bodyfbx' + '#')

    print(bodyfbx)
    print("="*100)
