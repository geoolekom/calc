import os
import scipy

COUNT = 1100
DATA_DIR = 'data/flow'

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datadir = os.path.join(current_dir, DATA_DIR)


for i in range(0, COUNT, 10):
    density_filename = os.path.join(datadir, 'density_{0:03d}.out'.format(i))
    density = scipy.genfromtxt(density_filename, delimiter="\t")

    temperature_filename = os.path.join(datadir, 'temperature_{0:03d}.out'.format(i))
    temperature = scipy.genfromtxt(density_filename, delimiter="\t")

    pressure_values = density[:, 2] * temperature[:, 2]
    pressure = scipy.dstack((density[:, 0], density[:, 1], pressure_values))[0]

    out_filename = os.path.join(datadir, 'pressure_{0:03d}.out'.format(i))
    out_file = open(out_filename, 'w')

    x_prev = None
    for x, y, p in pressure:
        if x_prev != x:
            if x_prev is not None:
                out_file.write('\n')
            x_prev = x
        out_file.write('{0}\t{1}\t{2}\n'.format(x, y, p))
    out_file.close()
