quantity = ARG1
count = ARG2
max_z = ARG3

set term gif animate delay 10 size 1000, 500
set xrange [-0.5:10.5]
set yrange [-0.5:5.5]
unset cbtics
unset key

set palette defined (0 "white", max_z "red")
set cbrange [0:1]
set cblabel "Density"
set view map

set output sprintf("data/%s_heatmap.gif", quantity)

do for [n = 0 : count - 1] {
    filename = sprintf("data/flow/%s_%03d.out", quantity, n)
    splot filename with pm3d
}
