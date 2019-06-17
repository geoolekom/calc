import os
import scipy
from scipy import stats

COUNT = 2400
DATA_DIR = 'data/06.05.19/8/flow'

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datadir = os.path.join(current_dir, DATA_DIR)

left = 34
right = 45

out_data = []
for i in range(0, COUNT, 10):
    filename = os.path.join(datadir, 'radius_{0:03d}.out'.format(i))
    data = scipy.genfromtxt(filename, delimiter="\t")
    x, y = data[:, 0][left:right], data[:, 1][left:right]

    a, _, _, _, err = stats.linregress(x, y)
    out_data.append([i, a, err])


out_filename = os.path.join(datadir, 'incline.out')
scipy.savetxt(out_filename, out_data, delimiter='\t', fmt='%f')
