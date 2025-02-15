import trimesh
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt



# basic visualization fucntions

def plot_panel_info(
    ax, panel_name, panel_svg_path_dict, stitch_dict,
    N_SAMPLES: int = 100,
):
    path = panel_svg_path_dict[panel_name][0]
    
    # boundary_points = np.array([path.point(t) for t in np.linspace(0, 1, N_SAMPLES)])
    # boundary_points = np.array([boundary_points.real, boundary_points.imag]).T
        
    # ax.plot(boundary_points[:, 0], boundary_points[:, 1], 'b-')
    ax.set_title(panel_name)
    ax.axis('equal')
    ax.grid(True)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(path)))

    for edge_idx, segment in enumerate(path):
        segment_points = np.array([
            [segment.point(t).real, segment.point(t).imag]
            for t in np.linspace(0, 1, N_SAMPLES)
        ])
        
        ax.plot(
            segment_points[:, 0],
            segment_points[:, 1] * -1,
            '-', color=colors[edge_idx]
        )
        
        segment_center = segment.point(0.5)
        segment_center = np.array([segment_center.real, segment_center.imag])
    
        has_stitch = False
    
        for stitch_idx, stitch_edges in stitch_dict.items():
            for edge_info in stitch_edges:
                if not isinstance(edge_info, dict):
                    # print(edge_info)
                    continue
                if edge_info['edge'] == edge_idx and edge_info['panel'] == panel_name:
                    has_stitch = True
                    ax.text(
                        segment_center[0],
                        -segment_center[1],
                        f"{stitch_idx}\n{edge_info['edge']}",
                        ha='center', va='center'
                    )

        if not has_stitch:
            ax.text(
                segment_center[0],
                -segment_center[1],
                f"no stitch,\n{edge_idx}",
                ha='center', va='center'
            )
            
            

def visualize_meshes_plotly(
    mesh_list,
    color_list=None,
    vertices_list = None,
    vertices_color_list = None,
    vertex_marker_size = 2,
    show_edges = True,
    edge_width = 2,
    show = True,
):
    # Pre-convert to list and load meshes once
    mesh_list = [mesh_list] if not isinstance(mesh_list, list) else mesh_list
    final_mesh_list = [trimesh.load(m) if isinstance(m, str) else m for m in mesh_list]
    
    color_list = color_list or ['lightgray'] * len(final_mesh_list)
    
    # Create all mesh traces at once
    mesh_traces = []
    edge_traces = []
    for mesh, color in zip(final_mesh_list, color_list):
        face_colors = mesh.visual.face_colors[:, :3] if hasattr(mesh.visual, 'face_colors') else None
        mesh_traces.append(go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            opacity=0.5,
            facecolor=face_colors,
            color=None if face_colors is not None else color
        ))
        
        if show_edges:
            edge_x = []
            edge_y = []
            edge_z = []
            vertices = mesh.vertices
            for edge in mesh.edges:
                edge_x.extend([vertices[edge[0], 0], vertices[edge[1], 0], None])
                edge_y.extend([vertices[edge[0], 1], vertices[edge[1], 1], None])
                edge_z.extend([vertices[edge[0], 2], vertices[edge[1], 2], None])
        
            edge_traces.append(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(
                    color = color if color is not None else 'red',
                    width=edge_width
                ),
                name='Edges'
            ))
    
    fig = go.Figure(data = mesh_traces + edge_traces)
    
    if vertices_list is not None and vertices_color_list is not None:
        for vertex, color in zip(vertices_list, vertices_color_list):
            fig.add_trace(go.Scatter3d(
                x=vertex[:, 0],
                y=vertex[:, 1],
                z=vertex[:, 2],
                mode='markers',
                marker=dict(size=vertex_marker_size, color=color, opacity=1),
                name='Vertices'
            ))
    fig.update_layout(
        scene=dict(aspectmode='data'),
        width=800,
        height=800,
        showlegend=False
    )
    
    if show:
        fig.show()
    return fig
    
    
    
     
def v_id_map(vertices): 
    v_map = [None] * len(vertices) 
    v_map[0] = 0 
    for i in range(1, len(vertices)): 
        if all(vertices[i - 1] == vertices[i]): 
            v_map[i] = v_map[i-1]   
        else: 
            v_map[i] = v_map[i-1] + 1 
    return v_map


import numpy as np
from scipy.spatial import cKDTree


def find_panel_boundary_vertices(mesh, panel_vertex_mask, threshold=1e-5):
    """
    Find boundary vertices for a given panel
    Args:
        mesh: trimesh.Trimesh
        panel_vertex_mask: boolean array indicating which vertices belong to the panel
        threshold: distance threshold for considering vertices as overlapping
    Returns:
        boundary_vertices: indices of boundary vertices
        overlapping_vertices: dict mapping vertex indices to lists of nearby vertex indices
    """
    # Get panel vertices and faces
    panel_vertices = mesh.vertices[panel_vertex_mask]
    panel_vertex_indices = np.where(panel_vertex_mask)[0]
    
    # Find boundary vertices (vertices that are connected to non-panel vertices)
    boundary_vertices = set()
    overlapping_vertices = {}
    
    # For each edge in the mesh
    for edge in mesh.edges:
        v1, v2 = edge
        v1_in_panel = panel_vertex_mask[v1]
        v2_in_panel = panel_vertex_mask[v2]
        
        # If exactly one vertex is in the panel, the edge vertex in the panel is a boundary
        if v1_in_panel != v2_in_panel:
            if v1_in_panel:
                boundary_vertices.add(v1)
            if v2_in_panel:
                boundary_vertices.add(v2)
    
    # Find overlapping vertices using KDTree
    tree = cKDTree(mesh.vertices)
    
    for idx in panel_vertex_indices:
        # Find all vertices within threshold distance
        nearby_points = tree.query_ball_point(mesh.vertices[idx], threshold)
        
        # Remove self from nearby points and filter to only include points not in current panel
        nearby_points = [p for p in nearby_points if p != idx and not panel_vertex_mask[p]]
        
        if nearby_points:
            overlapping_vertices[idx] = nearby_points
            
    return list(boundary_vertices), overlapping_vertices


def find_overlapping_vertices(mesh, threshold=0.001):
    """
    Find vertices that are within threshold distance of each other
    Args:
        mesh: trimesh.Trimesh - the mesh to analyze
        threshold: float - distance threshold for considering vertices as overlapping
    Returns:
        overlapping_vertices: dict mapping vertex indices to lists of nearby vertex indices
    """
    tree = cKDTree(mesh.vertices)
    overlapping_vertices = {}
    
    for vertex_idx in range(len(mesh.vertices)):
        # Find all vertices within threshold distance
        nearby_points = tree.query_ball_point(mesh.vertices[vertex_idx], threshold)
        
        # Remove self from nearby points
        nearby_points = [p for p in nearby_points if p != vertex_idx]
        
        if nearby_points:
            overlapping_vertices[vertex_idx] = nearby_points
            
    return overlapping_vertices




def filter_segmentation_map(mesh, segmentation_list):
    """
    Filter segmentation map by fixing isolated vertices based on their neighbors
    
    Args:
        mesh: trimesh.Trimesh object
        segmentation_list: list of segmentation labels for each vertex
    
    Returns:
        filtered_segmentation_list: list of filtered segmentation labels
    """
    # Convert to numpy array for easier manipulation
    segmentation_array = np.array(segmentation_list)
    filtered_segmentation = segmentation_array.copy()
    
    # Get vertex adjacency
    vertex_neighbors = mesh.vertex_neighbors
    
    # Iterate through vertices
    for vertex_idx in range(len(segmentation_array)):
        current_label = segmentation_array[vertex_idx]
        neighbors = vertex_neighbors[vertex_idx]
        
        if len(neighbors) == 0:
            continue
            
        # Get neighbor labels
        neighbor_labels = segmentation_array[neighbors]
        
        # Count occurrences of each label in neighbors
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        
        # If current vertex label doesn't match majority of neighbors
        if current_label not in neighbor_labels:
            # Replace with most common neighbor label
            filtered_segmentation[vertex_idx] = unique_labels[counts.argmax()]
            
    return filtered_segmentation.tolist()


import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import trimesh

def filter_segmentation_map_clusters(mesh, segmentation_list, threshold=30):
    """
    Refines the segmentation of a mesh by removing small clusters 
    and replacing them with surrounding dominant segments.
    
    Args:
        mesh: trimesh.Trimesh object representing the garment mesh.
        segmentation_list: List of segment labels for each vertex.
        threshold: Minimum size of a valid cluster. Smaller clusters are reassigned.
    
    Returns:
        A refined segmentation list.
    """
    segmentation_array = np.array(segmentation_list)  # Convert to NumPy array
    
    # Step 1: Build a graph of mesh connectivity
    G = nx.Graph()
    vertices = mesh.vertices
    faces = mesh.faces
    
    for face in faces:
        for i in range(3):
            G.add_edge(face[i], face[(i + 1) % 3])  # Connect face vertices
    
    # Step 2: Find connected components within each segmentation label
    label_to_components = {}
    for label in np.unique(segmentation_array):
        subgraph = G.subgraph([i for i, seg in enumerate(segmentation_array) if seg == label])
        components = list(nx.connected_components(subgraph))
        label_to_components[label] = components
    
    # Step 3: Identify small clusters and reassign them
    refined_segmentation = segmentation_array.copy()
    
    for label, components in label_to_components.items():
        for cluster in components:
            if len(cluster) < threshold:  # If cluster is too small
                # Find neighboring vertices with different segmentation
                neighboring_labels = []
                for vertex in cluster:
                    for neighbor in G.neighbors(vertex):
                        if refined_segmentation[neighbor] != label:
                            neighboring_labels.append(refined_segmentation[neighbor])
                
                if neighboring_labels:
                    # Assign the most frequent neighboring segmentation
                    most_common_label = max(set(neighboring_labels), key=neighboring_labels.count)
                    refined_segmentation[list(cluster)] = most_common_label
    
    return refined_segmentation.tolist()



import numpy as np
import trimesh
from collections import Counter

def reclassify_none_vertices(mesh, vertex_labels):
    """
    Reclassifies 'None' vertices based on neighboring vertex labels.
    
    Args:
    - mesh (trimesh.Trimesh): The 3D mesh object.
    - vertex_labels (list): A list of segmentation labels for each vertex. 'None' represents unclassified vertices.
    
    Returns:
    - Updated vertex_labels with no 'None' values.
    """
    
    # Convert to numpy array for easy indexing
    vertex_labels = np.array(vertex_labels, dtype=object)
    
    # Get adjacency information from mesh
    adjacency = mesh.vertex_neighbors

    # Identify indices of "None" vertices
    none_indices = np.where(vertex_labels == "None")[0]
    
    for idx in none_indices:
        neighbors = adjacency[idx]  # Get neighboring vertex indices
        neighbor_labels = [vertex_labels[n] for n in neighbors if vertex_labels[n] != "None"]

        if len(neighbor_labels) == 0:
            continue  # Skip if no valid neighbors exist
        
        # Majority voting among neighbors
        most_common_label, count = Counter(neighbor_labels).most_common(1)[0]
        vertex_labels[idx] = most_common_label  # Assign the most common neighbor class
    
    return vertex_labels.tolist()
