import os
import scipy

COUNT = 1000
DATA_DIR = 'data/18.04.19/flow 1 thin'

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datadir = os.path.join(current_dir, DATA_DIR)

for i in range(0, COUNT, 10):
    filename = os.path.join(datadir, 'radius_{0:03d}.out'.format(i))
    data = scipy.genfromtxt(filename, delimiter="\t")
    x, y = data[:, 0][62:102], data[:, 1][62:102]
    fp, residuals, rank, sv, rcond = scipy.polyfit(x, y, 1, full=True)
    print(i, fp[0], sep='\t')
