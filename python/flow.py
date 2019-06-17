import os
import scipy
import numpy
from matplotlib import pyplot

DATA_DIR = 'data/06.05.19/8/flow'

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datadir = os.path.join(current_dir, DATA_DIR)

flow_filename = os.path.join(datadir, 'flow.out')
flow = scipy.genfromtxt(flow_filename, delimiter="\t")

# shift = 5
# x = flow[shift // 2:-shift // 2, 0]
# values = (flow[shift:, 1] - flow[:-shift, 1]) / flow[shift // 2:-shift // 2, 1] / shift

x = flow[1:-1, 0]
ln_values = numpy.log(flow[:, 1])
# values = (ln_values[2:] + ln_values[:-2] - 2 * ln_values[1:-1])
values = (ln_values[2:] - ln_values[:-2]) / 2

flow_diff = scipy.dstack((x, values))[0]

out_filename = os.path.join(datadir, 'flow_diff_avg.out')
scipy.savetxt(out_filename, flow_diff, delimiter='\t', fmt='%f')

pyplot.plot(x, values, linestyle='None', marker='o')
# pyplot.plot(x, scipy.full_like(x, 0.001))
# pyplot.plot(x, scipy.full_like(x, -0.001))
pyplot.ylim(-0.005, 0.005)
pyplot.show()
