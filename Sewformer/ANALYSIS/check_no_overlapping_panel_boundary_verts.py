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
import networkx as nx

from env_constants import SEWFORMER_PROJ_ROOT, DATASET_ROOT, PYGARMENT_ROOT

sys.path.append(PYGARMENT_ROOT)

import pygarment as pyg


def find_boundary_vertices(mesh):
    """
    Identifies boundary vertices in a manifold mesh.
    
    - Boundary edges are edges that belong to only one face.
    - Boundary vertices are the endpoints of boundary edges.
    
    Args:
        mesh (trimesh.Trimesh): The input mesh object.
    
    Returns:
        np.array: Indices of boundary vertices.
    """
    edges, counts = np.unique(np.sort(mesh.edges, axis=1), axis=0, return_counts=True)
    boundary_edges = edges[counts == 1]
    boundary_vertices = np.unique(boundary_edges)

    return boundary_vertices

def disassemble_mesh_to_panels(mesh, segmentation_list):
    """
    Disassembles a garment mesh into separate panel meshes and splits the segmentation list.

    Args:
        mesh (trimesh.Trimesh): The input garment mesh.
        segmentation_list (list): List classifying each vertex.

    Returns:
        tuple: (list of trimesh.Trimesh, list of segmentation lists for each panel)
    """
    # Step 1: Create a graph from the mesh connectivity
    mesh_graph = nx.Graph()

    # Add edges based on mesh faces (each face connects three vertices)
    for face in mesh.faces:
        mesh_graph.add_edge(face[0], face[1])
        mesh_graph.add_edge(face[1], face[2])
        mesh_graph.add_edge(face[2], face[0])

    # Step 2: Find connected components (each should be an individual panel)
    connected_components = list(nx.connected_components(mesh_graph))
    print(f"Found {len(connected_components)} separate panel meshes.")

    # Step 3: Extract panel sub-meshes
    panel_meshes = []
    panel_segmentation_list = []
    garment_to_panel_idx_map_list = []
    panel_to_garment_idx_map_list = []
    for component in connected_components:
        panel_vertex_indices = np.array(list(component))  # Original indices
        
        # Step 3.1: Create a mapping from original to new local indices
        # index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(panel_vertex_indices)}
        garment_to_panel_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(panel_vertex_indices)}
        panel_to_garment_idx_map = {new_idx: old_idx for old_idx, new_idx in enumerate(panel_vertex_indices)}

        garment_to_panel_idx_map_list.append(garment_to_panel_idx_map)
        panel_to_garment_idx_map_list.append(panel_to_garment_idx_map)

        # Step 3.2: Extract the segmentation list for this panel
        this_panel_segmentation_list = [segmentation_list[idx] for idx in panel_vertex_indices]
        panel_segmentation_list.append(this_panel_segmentation_list)

        # Step 3.3: Get faces that only use the panel's vertices
        face_mask = np.isin(mesh.faces, panel_vertex_indices).all(axis=1)
        panel_faces = mesh.faces[face_mask]

        # Step 3.4: Reindex faces to local vertex indices
        reindexed_faces = np.vectorize(garment_to_panel_idx_map.get)(panel_faces)

        # Step 3.5: Create new mesh
        panel_mesh = trimesh.Trimesh(
            vertices=mesh.vertices[panel_vertex_indices],
            faces=reindexed_faces
        )
        panel_meshes.append(panel_mesh)

    return panel_meshes, panel_segmentation_list, garment_to_panel_idx_map_list, panel_to_garment_idx_map_list


def filter_fn(path):
    if not os.path.isdir(path):
        print(f"{path} is not a directory")
        return False
    if not os.path.exists(os.path.join(path, "static", "spec_config.json")):
        print(f"No spec_config.json in {path}")
        return False
    return True


if __name__ == "__main__":
    
    

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

            garment_name_list = list(map(
                lambda x : os.path.basename(x["spec"].replace("\\", "/")),
                spec_config.values()
            ))

            garment_dict = {}
            for garment_name in garment_name_list :
                spec_file_path = os.path.join(
                    combination_path, "static", f"{garment_name}_specification.json"
                )
                pattern = pyg.pattern.wrappers.VisPattern(spec_file_path)
                drawn_pattern_list = list(map(
                    lambda pannel_name : pattern._draw_a_panel(
                        pannel_name, apply_transform=False, fill=True
                    ),
                    pattern.panel_order()
                ))
                panel_svg_path_dict = {
                    panel_name : pattern._draw_a_panel(
                        panel_name, apply_transform=False, fill=True
                    )
                    for panel_name in pattern.panel_order()
                }

                stitch_dict = {
                    i : v for i, v in enumerate(pattern.pattern['stitches'])
                }
                
                mesh = trimesh.load_mesh(os.path.join(
                    combination_path, "static", f"{garment_name}_{garment_name}.obj",
                ), process=False)

                neighbor_idx_list = mesh.vertex_neighbors
                
                with open(
                    os.path.join(
                        combination_path, "static", f"{garment_name}_{garment_name}_segmentation.txt"
                    ),
                    "r"
                ) as f:
                    mesh_segmentation_list = list(map(
                        lambda x : x.strip(),
                        f.readlines()
                    ))
                    vertex_mask_dict = {}
                    for panel_name in panel_svg_path_dict.keys() :
                        vertex_mask_dict[panel_name] = np.array(list(map(
                            lambda x : x == panel_name,
                            mesh_segmentation_list
                        )))
                    vertex_mask_dict["stitch"] = np.array(list(map(
                        lambda x : x == "stitch",
                        mesh_segmentation_list
                    )))
                    vertex_mask_dict["None"] = np.array(list(map(
                        lambda x : x == "None",
                        mesh_segmentation_list
                    )))

                (
                    panel_mesh_list, panel_segmentation_list,
                    garment_to_panel_idx_map_list, panel_to_garment_idx_map_list
                ) = disassemble_mesh_to_panels(mesh, mesh_segmentation_list)
                
                panel_dict = {}
                for (
                    panel_mesh, panel_segmentation_list, 
                    garment_to_panel_idx_map, panel_to_garment_idx_map
                )in zip(
                    panel_mesh_list, panel_segmentation_list,
                    garment_to_panel_idx_map_list, panel_to_garment_idx_map_list
                ) :
                    panel_name_candidate_list, count_list = np.unique(panel_segmentation_list, return_counts=True)
                    panel_name = panel_name_candidate_list[np.argmax(count_list)]
                    
                    boundary_vert_idx_list = find_boundary_vertices(panel_mesh)
                    
                    
                    overlapping_boundary_vert_count = np.sum(
                        np.linalg.norm(
                            panel_mesh.vertices[boundary_vert_idx_list[:-1]] - panel_mesh.vertices[boundary_vert_idx_list[1:]],
                            axis=1
                        ) < 1e-6
                    )
                    if overlapping_boundary_vert_count > 0 :
                        print(combination_path)
                        print(garment_name)
                        print(panel_name, overlapping_boundary_vert_count)
                        print()
                            
                    panel_dict[panel_name] = {
                        "panel_mesh" : panel_mesh,
                        "boundary_vert_idx_list" : boundary_vert_idx_list,
                        "garment_to_panel_idx_map" : garment_to_panel_idx_map,
                        "panel_to_garment_idx_map" : panel_to_garment_idx_map,
                    }
                    
                garment_dict[garment_name] = {
                    "panel_svg_path_dict" : panel_svg_path_dict,
                    "stitch_dict" : stitch_dict,
                    "mesh" : mesh,
                    "panel_dict" : panel_dict,
                }
            
        except Exception as e:
            print(f"Error in {combination_path}: {e}")
            traceback.print_exc()
            print()
            continue
