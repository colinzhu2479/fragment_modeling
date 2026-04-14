"""
Model Loader and Prediction Module.

With the system's graph representation, exploration of new regions of data space 
(the potential energy configuration space) is carried out by constructing a series 
of localized, lower-dimensional fragment (subsystem) spaces. 
This module handles loading these localized trained models from disk and executing 
predictions for both fragment energies and atomic forces.
"""

import numpy as np
import scipy
import tensorflow as tf
from scipy import spatial
from scipy.spatial import distance
from tensorflow import keras
import os
from typing import List, Dict, Tuple, Any, Optional

from fragment_modeling.fragment_transform import *


def load_energy_model(num_atom_list: List[int], model_path_root: str) -> Tuple[Dict[str, keras.Model], Dict[str, np.ndarray]]:
    """
    Loads primary and secondary energy prediction models alongside their normalization parameters.
    
    Args:
        num_atom_list (List[int]): A list of atom counts defining the fragment sizes to load.
        model_path_root (str): The root directory where model files are stored.
        
    Returns:
        Tuple[Dict[str, keras.Model], Dict[str, np.ndarray]]: 
            - model_dict: A dictionary mapping identifiers to keras Models.
            - param_dict: A dictionary mapping identifiers to loaded normalization parameter arrays.
    """
    tf.get_logger().setLevel('ERROR')
    model_dict = {}
    param_dict = {}
    
    for ll in num_atom_list:
        l = ll
        model_path = os.path.join(model_path_root, str(ll), '')
        
        # Determine the saved model format
        if os.path.exists(model_path + f"{l}_primary.h5"):
            form = '.h5'
        elif os.path.exists(model_path + f"{l}_primary.tf"):
            form = '.tf'
        else:
            pass # Fails silently if not found based on original code logic
            
        # Load Primary Model
        model_dict[f"{ll}p"] = keras.models.load_model(model_path + f"{l}_primary{form}")
        
        # Load Secondary (Delta/Error) Model
        model_dict[f"{ll}s"] = keras.models.load_model(model_path + f"{l}_secondary{form}")
        
        # Load Normalization Parameters
        param_dict[f"{ll}px"] = np.loadtxt(model_path + f"{l}_p_x.txt")
        param_dict[f"{ll}py"] = np.loadtxt(model_path + f"{l}_p_y.txt")
        param_dict[f"{ll}sy"] = np.loadtxt(model_path + f"{l}_s_y.txt")
        print(f'{ll} energy model loaded.')
        
    return model_dict, param_dict


def load_force_model(frag_list: List[str], target_direction: str, model_path_root: str) -> Tuple[Dict[str, keras.Model], Dict[str, np.ndarray]]:
    """
    Loads directional (x, y, or z) force prediction models and parameters.
    
    Args:
        frag_list (List[str]): List of fragment identifiers to load models for.
        target_direction (str): The Cartesian direction to load ('x', 'y', or 'z').
        model_path_root (str): The root directory where model files are stored.
        
    Returns:
        Tuple[Dict[str, keras.Model], Dict[str, np.ndarray]]: Model and parameter dictionaries.
    """
    tf.get_logger().setLevel('ERROR')
    model_dict = {}
    param_dict = {}
    
    for ll in frag_list:
        l = ll 
        model_path = os.path.join(model_path_root, str(ll), target_direction, '')
        
        # Determine the saved model format
        if os.path.exists(model_path + f"{l}_primary.h5"):
            format_type = '.h5'
        elif os.path.exists(model_path + f"{l}_primary.tf"):
            format_type = '.tf'
        else:
            print('Model not found. ' + model_path + f"{l}_primary")
            
        model_dict[f"{target_direction}{ll}p"] = keras.models.load_model(model_path + f"{l}_primary{format_type}")
        model_dict[f"{target_direction}{ll}s"] = keras.models.load_model(model_path + f"{l}_secondary{format_type}")
        
        param_dict[f"{target_direction}{ll}px"] = np.loadtxt(model_path + f"{l}_p_x.txt")
        param_dict[f"{target_direction}{ll}py"] = np.loadtxt(model_path + f"{l}_p_y.txt")
        param_dict[f"{target_direction}{ll}sy"] = np.loadtxt(model_path + f"{l}_s_y.txt")
        print(f'{ll} {target_direction} force model loaded.')
        
    return model_dict, param_dict


def predict_energy(xyz: List[float], partition_index: Any, model_dict: Dict[str, keras.Model], param_dict: Dict[str, np.ndarray], model_name: str) -> np.ndarray:
    """
    Predicts the energy of a localized configuration by applying pre-trained primary and secondary models.
    
    Args:
        xyz (List[float]): Distances or spatial input data for the target fragment.
        partition_index (Any): Index mapping for this specific fragment within the entire structure.
        model_dict (Dict): Dictionary containing loaded keras models.
        param_dict (Dict): Dictionary containing normalization ranges and minimums.
        model_name (str): The identifier key for the target model.
        
    Returns:
        np.ndarray: Predicted physical energy array.
    """
    p_dis = np.array(xyz)

    # Prepare and scale the input data for the model
    x_range, xmin = param_dict[f"{model_name}px"][0], param_dict[f"{model_name}px"][1]
    p_dis = ((p_dis - xmin) / x_range).reshape(1, np.size(p_dis))

    # Load energy scaling bounds
    y_range, y_min = param_dict[f"{model_name}py"][0], param_dict[f"{model_name}py"][1]
    e_range, e_min = param_dict[f"{model_name}sy"][0], param_dict[f"{model_name}sy"][1]

    # Combine primary prediction with secondary (correction) prediction and scale back to physical units
    p_e = model_dict[f"{model_name}p"](p_dis) * y_range + y_min + \
          (model_dict[f"{model_name}s"](p_dis) * e_range + e_min) / 627.503

    return p_e.numpy()


def predict_force(
    atomic_num: List[int], xyz: List[List[float]], partition_index: List[List[int]], 
    model_dict: Dict[str, keras.Model], param_dict: Dict[str, np.ndarray], 
    fragment_name_input: Optional[str], real: Optional[Any] = None
) -> np.ndarray:
    """
    Predicts directional atomic forces over molecular subsets (partitions) using geometric transformations 
    and lower-dimensional trained ML models.
    
    Args:
        atomic_num (List[int]): Full list of atomic numbers in the system.
        xyz (List[List[float]]): Cartesian coordinates for the full system.
        partition_index (List[List[int]]): Node groupings defining the subsystems/fragments.
        model_dict (Dict): The loaded keras force models.
        param_dict (Dict): The loaded scaling parameter definitions.
        fragment_name_input (Optional[str]): Explicit name/identifier of the fragment; if None, falls back to atom count.
        real (Optional[Any]): Unused placeholder for real physical values mapping.
        
    Returns:
        np.ndarray: Matrix of predicted force vectors matching the original structural geometry.
    """
    direction = ['x', 'y', 'z']
    predicted_energy_list = np.zeros(len(partition_index), dtype='object')
    n = 0

    for p in partition_index:  # Iterate over every structural simplex / fragment
        num_atom = len(p)
        num_dis = int(num_atom * (num_atom - 1) / 2)
        p_dis = np.zeros(num_dis)

        partition_ar = np.array(atomic_num)[p]
        xyz_t = np.array(xyz)[p]

        # Force related transformation specific setup
        force = np.zeros([len(partition_ar), 3])
        ref = np.ones([len(partition_ar), 3]) * 3
        atomic_mass = atom_mass_mapping(partition_ar)
        
        # Need global permutation information
        reconstructed_force = np.zeros([3, len(partition_ar)])

        # Generate transformed features for all 3 dimensions invariant frames
        for c, ii in enumerate(global_transform_fragment(xyz_t, force, partition_ar, atomic_mass, ref)):
            xyz_tt, force, partition_ar, v, order = ii
            target_direction = direction[c]
            p_dis = p_dis.reshape(-1)
            
            # Formulate the distance matrix representing the lower-dimensional subsystem space
            dis_matrix = distance.cdist(xyz_tt, xyz_tt)
            k = 0
            for i in range(len(dis_matrix) - 1):
                for j in range(i, len(dis_matrix) - 1):
                    p_dis[k] = dis_matrix[i][j + 1]
                    k += 1

            fragment_name = fragment_name_input if fragment_name_input is not None else str(num_atom)

            # Retrieve parameters and normalize input for the simplex
            x_range, xmin = param_dict[f"{target_direction}{fragment_name}px"][0], param_dict[f"{target_direction}{fragment_name}px"][1]
            p_dis = ((p_dis - xmin) / x_range).reshape(1, num_dis)

            y_range, y_min = param_dict[f"{target_direction}{fragment_name}py"][0], param_dict[f"{target_direction}{fragment_name}py"][1]
            e_range, e_min = param_dict[f"{target_direction}{fragment_name}sy"][0], param_dict[f"{target_direction}{fragment_name}sy"][1]

            # Execute Model Predictions
            p_e = model_dict[f"{target_direction}{fragment_name}p"](p_dis) * y_range + y_min + \
                  (model_dict[f"{target_direction}{fragment_name}s"](p_dis) * e_range + e_min) / 627.503
                  
            # Reconstruct correctly sorted outputs
            reconstructed_force[c] = p_e[0].numpy()[np.argsort(order)]

        # Apply inverse transform matrix to map local predicted forces back to global cartesian space
        inv_v = np.linalg.inv(v)
        predicted_energy_list[n] = np.matmul(inv_v, reconstructed_force).T
        n += 1
        
    return predicted_energy_list
