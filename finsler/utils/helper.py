import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch


def pickle_save(data, folder_path, file_name):
    data_path = folder_path
    if not os.path.exists(data_path):
        print("creating folder for experiments: {}".format(data_path))
        os.mkdir(data_path)
    with open(os.path.join(data_path, file_name), "wb") as fw:
        pickle.dump(data, fw)


def pickle_load(folder_path, file_name):
    data_path = os.path.join(folder_path, file_name)
    with open(data_path, "rb") as fr:
        data = pickle.load(fr)
    return data


def psd_matrix(eigval):
    """Create a psd matrix based on its eigenvalues"""
    dim = len(eigval)
    D = np.diag(eigval)
    P, _ = scipy.linalg.qr(np.random.rand(dim, dim))
    return P.T @ D @ P


def is_pos_def(x):
    "check if matrix is pd"
    return np.all(np.linalg.eigvals(x) > 0)


def to_np(x):
    return np.squeeze(x.detach().numpy())


def to_torch(x):
    return torch.from_numpy(x)


def create_folder(pathname):
    if not os.path.isdir(pathname):
        os.makedirs(pathname)


def create_filepath(pathname, filename):
    filename, extsn = filename.split(".")
    filenum = 0
    filepath = str(pathname + "/" + filename + "_")
    while os.path.exists(filepath + str(filenum) + "." + extsn):
        filenum += 1
    return filepath + str(filenum) + "." + extsn
