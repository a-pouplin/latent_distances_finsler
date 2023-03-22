import os

import numpy as np

from finsler.utils.helper import create_filepath, create_folder

folderpath = os.path.abspath("debug/debug")
create_folder(folderpath)
filename = "dummymatrix.npy"
filepath = create_filepath(folderpath, filename)
matrix = np.random.randn(4, 4)
np.save(filepath, matrix)
