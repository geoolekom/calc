quantity = ARG1
count = ARG2
max_z = ARG3

set term gif animate delay 10 size 800, 800
set xrange [-0.5:10.5]
set yrange [-0.5:5.5]
set zrange [0:max_z]

set output sprintf("data/%s_3d.gif", quantity)

do for [n = 0 : count - 1] {
    filename = sprintf("data/flow/%s_%03d.out", quantity, n)
    splot filename with lines
}
