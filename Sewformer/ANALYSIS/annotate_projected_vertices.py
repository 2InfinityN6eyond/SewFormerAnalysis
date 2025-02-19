import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
# sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from glob import glob
import numpy as np
import json
import pickle
import trimesh
from tqdm import tqdm
from PIL import Image
from ANALYSIS.analysis_utils import (
    plot_panel_info,
    visualize_meshes_plotly,
    filter_segmentation_map,
    filter_segmentation_map_clusters,
    is_clockwise,
)
import traceback

from env_constants import SEWFORMER_PROJ_ROOT, DATASET_ROOT, PYGARMENT_ROOT

sys.path.append(PYGARMENT_ROOT)

import pygarment as pyg


def filter_fn(path):
    if not os.path.isdir(path):
        print(f"{path} is not a directory")
        return False
    if not os.path.exists(os.path.join(path, "static", "spec_config.json")):
        print(f"No spec_config.json in {path}")
        return False

    try:
        combination_path = path
        spec_config_path = os.path.join(combination_path, "static", "spec_config.json")
        with open(spec_config_path, "r", encoding='utf-8') as f:
            spec_config = json.load(f)

        combination_garment_name_list = list(map(
            lambda x : os.path.basename(x["spec"].replace("\\", "/")),
            spec_config.values()
        ))

        static_camera_dict = {}
        for camera_path in sorted(glob(os.path.join(combination_path, "static", "*cam_pos.json"))):
            try:
                with open(camera_path, "r", encoding='utf-8') as f:
                    camera_data = json.load(f)
                camera_name = os.path.basename(camera_path).replace("_cam_pos.json", "")
                static_camera_dict[camera_name] = camera_data
            except UnicodeDecodeError:
                print(f"Warning: Could not read camera file {camera_path} - skipping")
                continue

        # if already annotated, skip
        if (
            len(glob(
                os.path.join(combination_path, "static", "*_visibility_mask.pkl")
            )) == len(combination_garment_name_list) * len(static_camera_dict)
        ) and (
            len(glob(
                os.path.join(combination_path, "static", "*_pixel_coords.pkl")
            )) == len(combination_garment_name_list) * len(static_camera_dict)
        ):
            return False
            
        return True
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return False




if __name__ == "__main__":
    
    import os
    import platform
    if platform.system() == 'Linux':
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ['MESA_GL_VERSION_OVERRIDE'] = '4.1' 
        os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '410'

    import pyrender
    
    import smplx
    import torch

    SMPLH_PATH = os.path.join(
        SEWFORMER_PROJ_ROOT, "Sewformer", "assets",
    )

    # Create SMPL-H model
    model = smplx.create(
        model_path=SMPLH_PATH,
        model_type='smplh',  # Specifically use SMPL-H
        ext='pkl',
        gender='female',
        use_pca=False,  # Important: disable PCA for hand poses
        batch_size=1,
    )
    
    

    raw_combination_path_list = sorted(glob(os.path.join(DATASET_ROOT, "sewfactory", "*")))
    combination_path_list = []
    for garment_path in tqdm(raw_combination_path_list):
        if filter_fn(garment_path):
            combination_path_list.append(garment_path)
    
    
    for combination_path in tqdm(combination_path_list) :
        try :


            spec_config_path = os.path.join(combination_path, "static", "spec_config.json")
            with open(spec_config_path, "r") as f:
                spec_config = json.load(f)

            combination_garment_name_list = list(map(
                lambda x : os.path.basename(x["spec"].replace("\\", "/")),
                spec_config.values()
            ))


            static_camera_dict = {}
            for camera_path in sorted(glob(os.path.join(combination_path, "static", "*cam_pos.json"))):
                with open(camera_path, "r") as f:
                    camera_data = json.load(f)
                camera_name = os.path.basename(camera_path).replace("_cam_pos.json", "")
                static_camera_dict[camera_name] = camera_data
                



            with open(os.path.join(combination_path, "static", "static__body_info.json"), "r") as f:
                static_body_data = json.load(f)
                betas = torch.tensor(static_body_data['shape'], dtype=torch.float32).unsqueeze(0)
                pose = torch.tensor(
                    np.deg2rad(static_body_data['pose']),
                    dtype=torch.float32
                ).unsqueeze(0)  # Shape: (1, 52, 3)
                transl = torch.tensor(static_body_data['trans'], dtype=torch.float32).unsqueeze(0)

                body_pose = pose[0, 1:22].reshape(1, -1)  # Body joints (excluding global orientation)
                left_hand_pose = pose[0, 22:37].reshape(1, -1)  # Left hand joints
                right_hand_pose = pose[0, 37:52].reshape(1, -1)  # Right hand joints
                global_orient = pose[0, 0].unsqueeze(0)  # Global orientation

                # Get body mesh
                output = model(
                    betas=betas,
                    body_pose=body_pose,
                    global_orient=global_orient,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    # transl=transl
                )
                vertices = output.vertices.detach().numpy()[0]
                faces = model.faces

                # SCALE = 2 * transl.numpy()[0, 1] / (vertices[:, 1].max() - vertices[:, 1].min())
                SCALE = 95
                Z_OFFSET = -2.2
                Y_OFFSET = 21.2

                static_body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                static_body_mesh.apply_scale(SCALE)

                static_body_mesh.vertices += transl.numpy()
                # static_body_mesh.vertices[:, 1] -= static_body_mesh.vertices[:, 1].min()
                static_body_mesh.vertices[:, 2] += Z_OFFSET
                static_body_mesh.vertices[:, 1] += Y_OFFSET

            static_garment_dict = {}
            for garment_name in combination_garment_name_list :
                mesh_path = os.path.join(
                    combination_path, "static", f"{garment_name}_{garment_name}.obj"
                )
                mesh = trimesh.load_mesh(mesh_path, process=False)
                # with open(
                #     os.path.join(combination_path, "static", f"{garment_name}_{garment_name}_segmentation_filtered.txt"),
                #     "r"
                # ) as f:
                #     mesh_segmentation_list = list(map(
                #         lambda x : x.strip(),
                #         f.readlines()
                #     ))
                texture_image = Image.open(os.path.join(
                    combination_path, "static", f"{garment_name}_{garment_name}_uv.png"
                ))
                mesh.visual = trimesh.visual.TextureVisuals(
                    mesh.visual.uv,
                    texture_image
                )
                mesh.visual.material.doubleSided = True
                
                static_garment_dict[garment_name] = {
                    "mesh" : mesh,
                    # "mesh_segmentation_list" : mesh_segmentation_list
                }

            static_cam_name_list = list(static_camera_dict.keys())
            for static_cam_name in static_cam_name_list :
                cam = static_camera_dict[static_cam_name]
                cam_T = np.array(cam["cam_T"])
                cam_R = np.array(cam["cam_R"])
                cam_pose = np.eye(4)
                cam_pose[:3, 3] = cam_T
                cam_pose[:3, :3] = cam_R
                cam_direction = -cam_R[:, 2]

                body_material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=(0.0, 0.0, 0.0, 1.0),  # RGB color, Alpha
                    metallicFactor=0.658,  # Range: [0.0, 1.0]
                    roughnessFactor=0.5  # Range: [0.0, 1.0]
                )
                pyrender_body_mesh = pyrender.Mesh.from_trimesh(
                    static_body_mesh, material=body_material
                )

                pyrender_garment_mesh_list = [
                    pyrender.Mesh.from_trimesh(mesh_dict["mesh"]) for mesh_dict in static_garment_dict.values()
                ]

                cam_K = np.array(cam["cam_K"])
                fx = cam_K[0, 0]    
                fy = cam_K[1, 1]
                cx = cam_K[0, 2]
                cy = cam_K[1, 2]

                pyrender_cam = pyrender.PerspectiveCamera(
                    yfov = (
                        np.rad2deg(np.arctan(cy / fx)) * 2 * 4.0017
                    )
                )
                scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0])
                scene.add(pyrender_body_mesh)
                for pyrender_garment_mesh in pyrender_garment_mesh_list:
                    scene.add(pyrender_garment_mesh)
                scene.add(pyrender_cam, pose=cam_pose)

                camera_node = list(filter(
                    lambda x : x.camera is not None,
                    scene.get_nodes()
                ))[-1]
                intensity = 80.
                light_positions = [
                    np.array([1.60614, 1.5341, 1.23701]),
                    np.array([1.31844, 1.92831, -2.52238]),
                    np.array([-2.80522, 1.2594, 2.34624]),
                    np.array([0.160261, 1.81789, 3.52215]),
                    np.array([-2.65752, 1.41194, -1.26328])
                ]
                light_colors = [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0]
                ]

                for i in range(5):
                    light = pyrender.PointLight(color=light_colors[i], intensity=intensity)
                    light_pose = np.eye(4)
                    light_pose[:3, 3] = light_positions[i]
                    scene.add(light, pose=light_pose)

                r = pyrender.OffscreenRenderer(
                    viewport_width=1024, viewport_height=1024
                )
                flags = pyrender.RenderFlags.SKIP_CULL_FACES
                color, depth = r.render(scene, flags=flags)
                r.delete()

                # calculate visible vertices
                view_matrix = np.linalg.inv(scene.get_pose(camera_node))
                proj_matrix = camera_node.camera.get_projection_matrix(1024, 1024)

                for garment_name, garment_mesh in static_garment_dict.items() :
                    vertices_homog = np.hstack([
                        garment_mesh["mesh"].vertices,
                        np.ones((garment_mesh["mesh"].vertices.shape[0], 1))
                    ])

                    view_proj = proj_matrix @ view_matrix
                    projected = vertices_homog @ view_proj.T

                    z_coords = projected[:, 2].copy()
                    projected = projected[:, :3] / projected[:, 3:4]

                    pixel_coords = np.zeros((projected.shape[0], 2))
                    pixel_coords[:, 0] = (projected[:, 0] + 1.0) * 1024 / 2.0
                    pixel_coords[:, 1] = 1024 - (projected[:, 1] + 1.0) * 1024 / 2.0

                    px = np.clip(pixel_coords[:, 0].astype(int), 0, 1024 - 1)
                    py = np.clip(pixel_coords[:, 1].astype(int), 0, 1024 - 1)

                    THRESHOLD = -0.5
                    visibility_mask = (z_coords > 0) & \
                                (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < 1024) & \
                                (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < 1024) & \
                                (z_coords + THRESHOLD < depth[py, px])
                                
                                
                    garment_mesh["visibility_mask"] = visibility_mask
                    garment_mesh["pixel_coords"] = pixel_coords
                    
                    # static_img_path = os.path.join(
                    #     combination_path, "static", f"{static_cam_name.replace('cam_pos.json', '.png')}"
                    # )
                    with open(os.path.join(
                        combination_path, "static",
                        f"{static_cam_name.replace('cam_pos.json', '')}_{garment_name}_{garment_name}_visibility_mask.pkl"
                    ), "wb") as f:
                        pickle.dump(visibility_mask, f)
                    with open(os.path.join(
                        combination_path, "static",
                        f"{static_cam_name.replace('cam_pos.json', '')}_{garment_name}_{garment_name}_pixel_coords.pkl"
                    ), "wb") as f:
                        pickle.dump(pixel_coords, f)
        except Exception as e:
            print(f"Error in {combination_path}: {e}")
            traceback.print_exc()
            print()
            continue
