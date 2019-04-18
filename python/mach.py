import os
import scipy

COUNT = 1000
DATA_DIR = 'data/17.04.19/flow 5 HR'

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datadir = os.path.join(current_dir, DATA_DIR)


for i in range(0, COUNT, 10):
    temperature = os.path.join(datadir, 'temperature_{0:03d}.out'.format(i))
    t_data = scipy.genfromtxt(temperature, delimiter="\t")
    t_data[:, 2] = (t_data[:, 2] + 1) / 3

    out_filename = os.path.join(datadir, 't_fixed_{0:03d}.out'.format(i))
    out_file = open(out_filename, 'w')
    x_prev = None
    for x, y, t in t_data:
        if x_prev != x:
            if x_prev is not None:
                out_file.write('\n')
            x_prev = x
        out_file.write('{0}\t{1}\t{2}\n'.format(x, y, t))
    out_file.close()

    # scipy.savetxt(os.path.join(datadir, 't_fixed_{0:03d}.out'.format(i)), t_data, delimiter='\t', fmt='%f')

for i in range(0, COUNT, 10):
    temperature = os.path.join(datadir, 't_fixed_{0:03d}.out'.format(i))
    t_data = scipy.genfromtxt(temperature, delimiter="\t")
    t_data = t_data[::51]

    velocity = os.path.join(datadir, 'velocity_{0:03d}.out'.format(i))
    v_data = scipy.genfromtxt(velocity, delimiter="\t")
    v_data = v_data[::51]

    mach_values = v_data[:, 2] / scipy.sqrt(5 * t_data[:, 2] / 3)

    mach_data = scipy.dstack((t_data[:, 0], mach_values))[0]

    scipy.savetxt(os.path.join(datadir, 'mach_fixed_{0:03d}.out'.format(i)), mach_data, delimiter='\t')
