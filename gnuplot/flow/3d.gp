quantity = ARG1
count = ARG2
max_z = ARG3
step = ARG4
if (!exists("step")) step = 1

set term gif animate delay 10 size 1000, 1000
set xrange [0:100]
set yrange [0:25]
set zrange [0:max_z]

set output sprintf("data/%s_3d.gif", quantity)

do for [n = 0 : count - 1 : step] {
    filename = sprintf("data/flow/%s_%03d.out", quantity, n)
    splot filename with lines
}
