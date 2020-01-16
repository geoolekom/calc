quantity = ARG1
count = ARG2
max_z = ARG3
step = ARG4
if (!exists("step")) step = 1

set term gif animate delay 10 size 750, 500
set xrange [25:50]
set yrange [0:25]
unset cbtics
unset key

set palette defined (0 "white", max_z "red")
set cbrange [0:max_z]
set view map

dir = "data/06.05.19/5"

set output sprintf("%s/%s_heatmap.gif", dir, quantity)

do for [n = 0 : count - 1 : step] {
    filename = sprintf("%s/flow/data_%03d.out", dir, n)
    splot filename using 1:2:3 with pm3d title ""
}
