import os
import scipy

COUNT = 1000
DATA_DIR = 'data/flow'
X = 24
DY = 26

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datadir = os.path.join(current_dir, DATA_DIR)


for i in range(0, COUNT, 10):
    temperature = os.path.join(datadir, 'density_{0:03d}.out'.format(i))
    d_data = scipy.genfromtxt(temperature, delimiter="\t")
    dx_data = d_data[X * DY: (X + 1) * DY]

    out_filename = os.path.join(datadir, 'density_{k}_{0:03d}.out'.format(i, k=X))
    scipy.savetxt(out_filename, dx_data[:, 1:], delimiter='\t', fmt='%f')
