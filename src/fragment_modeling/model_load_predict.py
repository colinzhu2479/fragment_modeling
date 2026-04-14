import numpy as np
import scipy
import tensorflow as tf

from scipy import spatial
from tensorflow import keras
import os

def load_energy_model(num_atom_list, model_path_root):
    tf.get_logger().setLevel('ERROR')
    model_dict = dict({})
    param_dict = dict({})
    for ll in num_atom_list:
        l = ll  # str(ll.count('-'))
        model_path = model_path_root + str(ll) + '/'
        if os.path.exists(model_path + "%s_primary.h5" % str(l)):
            form = '.h5'
        elif os.path.exists(model_path + "%s_primary.tf" % str(l)):
            form = '.tf'
        else:
            pass
        model = keras.models.load_model(model_path + "%s_primary" % str(l) + form)

        model_dict["%s" % str(ll) + "p"] = model
        model = keras.models.load_model(model_path + "%s_secondary" % str(l) + form)

        model_dict["%s" % str(ll) + "s"] = model
        param_dict["%s" % str(ll) + "px"] = np.loadtxt(model_path + "%s_p_x.txt" % str(l))
        param_dict["%s" % str(ll) + "py"] = np.loadtxt(model_path + "%s_p_y.txt" % str(l))
        param_dict["%s" % str(ll) + "sy"] = np.loadtxt(model_path + "%s_s_y.txt" % str(l))
        print(f'{ll} energy model loaded.')
    return model_dict, param_dict


def load_force_model(frag_list, target_direction, model_path_root):
    tf.get_logger().setLevel('ERROR')
    model_dict = dict({})
    param_dict = dict({})
    for ll in frag_list:
        l = ll  # str(ll.count('-'))
        model_path = model_path_root + str(ll) + '/' + target_direction + '/'
        if os.path.exists(model_path + "%s_primary.h5" % str(l)):
            format_type = '.h5'
        elif os.path.exists(model_path + "%s_primary.tf" % str(l)):
            format_type = '.tf'
        else:
            print('Model not found. ' + model_path + "%s_primary" % str(l))
        model = keras.models.load_model(model_path + "%s_primary" % str(l) + format_type)
        model_dict[target_direction + "%s" % str(ll) + "p"] = model
        model = keras.models.load_model(model_path + "%s_secondary" % str(l) + format_type)
        model_dict[target_direction + "%s" % str(ll) + "s"] = model
        param_dict[target_direction + "%s" % str(ll) + "px"] = np.loadtxt(model_path + "%s_p_x.txt" % str(l))
        param_dict[target_direction + "%s" % str(ll) + "py"] = np.loadtxt(model_path + "%s_p_y.txt" % str(l))
        param_dict[target_direction + "%s" % str(ll) + "sy"] = np.loadtxt(model_path + "%s_s_y.txt" % str(l))
        print(f'{ll} {target_direction} force model loaded.')
    return model_dict, param_dict


def predict_energy(xyz, partition_index, model_dict, param_dict, model_name):
    p_dis = np.array(xyz)

    ####predict energy from models for 1 simplex
    x_range = param_dict["%s" % str(model_name) + "px"][0]
    xmin = param_dict["%s" % str(model_name) + "px"][1]
    p_dis = ((p_dis - xmin) / x_range).reshape(1, np.size(p_dis))

    y_range = param_dict["%s" % str(model_name) + "py"][0]
    y_min = param_dict["%s" % str(model_name) + "py"][1]

    e_range = param_dict["%s" % str(model_name) + "sy"][0]
    e_min = param_dict["%s" % str(model_name) + "sy"][1]

    p_e = model_dict["%s" % str(model_name) + "p"](p_dis) * y_range + y_min + \
          (model_dict["%s" % str(model_name) + "s"](p_dis) * e_range + e_min) / 627.503

    return p_e.numpy()


def predict_force(atomic_num, xyz, partition_index, model_dict, param_dict, fragment_name_input, real=None):

    direction = ['x', 'y', 'z']
    predicted_energy_list = np.zeros(len(partition_index), dtype='object')
    n = 0

    for p in partition_index:  ### p every simplex

        num_atom = len(p)
        num_dis = int(num_atom * (num_atom - 1) / 2)
        p_dis = np.zeros(num_dis)

        partition_ar = np.array(atomic_num)[p]

        xyz_t = np.array(xyz)[p]

        """force related transformation specifically"""
        force = np.zeros([len(partition_ar), 3])
        ref = np.ones([len(partition_ar), 3]) * 3
        atomic_mass = atom_mass_mapping(partition_ar)
        '''need a global permutation information'''
        reconstructed_force = np.zeros([3, len(partition_ar)])

        for c, ii in enumerate(fragment_transform.global_transform_fragment(xyz_t, force, partition_ar,
                                                                            atomic_mass, ref)):

            xyz_tt, force, partition_ar, v, order = ii
            target_direction = direction[c]
            p_dis = p_dis.reshape(-1)
            dis_matrix = distance.cdist(xyz_tt, xyz_tt)
            k = 0
            for i in range(len(dis_matrix) - 1):
                for j in range(i, len(dis_matrix) - 1):
                    p_dis[k] = dis_matrix[i][j + 1]
                    k += 1

            if fragment_name_input is None:
                fragment_name = str(num_atom)
            else:
                fragment_name = fragment_name_input

            ####predict energy from models for 1 simplex
            x_range = param_dict[target_direction + "%s" % str(fragment_name) + "px"][0]
            xmin = param_dict[target_direction + "%s" % str(fragment_name) + "px"][1]
            p_dis = ((p_dis - xmin) / x_range).reshape(1, num_dis)

            y_range = param_dict[target_direction + "%s" % str(fragment_name) + "py"][0]
            y_min = param_dict[target_direction + "%s" % str(fragment_name) + "py"][1]

            e_range = param_dict[target_direction + "%s" % str(fragment_name) + "sy"][0]
            e_min = param_dict[target_direction + "%s" % str(fragment_name) + "sy"][1]

            p_e = model_dict[target_direction + "%s" % str(fragment_name) + "p"](p_dis) * y_range + y_min + \
                  (model_dict[target_direction + "%s" % str(fragment_name) + "s"](p_dis) * e_range + e_min) / 627.503
            reconstructed_force[c] = p_e[0].numpy()[np.argsort(order)]

        inv_v = np.linalg.inv(v)
        dir_ord = direction_order(target_direction)
        predicted_energy_list[n] = np.matmul(inv_v, reconstructed_force).T
        n += 1
    return predicted_energy_list
