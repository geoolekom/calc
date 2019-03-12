quantity = ARG1
count = ARG2
max_z = ARG3
step = ARG4
if (!exists("step")) step = 1

set term gif animate delay 10 size 2000, 1000
set xrange [6:18]
set yrange [0:5]
unset cbtics
unset key

set palette defined (0 "white", max_z "red")
set cbrange [0:max_z]
set view map

set output sprintf("data/%s_heatmap.gif", quantity)

do for [n = 0 : count - 1 : step] {
    filename = sprintf("data/flow/%s_%03d.out", quantity, n)
    splot filename with pm3d
}
