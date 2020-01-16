quantity = ARG1
n = ARG2
max_z = ARG3

set term pngcairo enhanced
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0

set xrange [25:50]
set yrange [0:25]
unset cbtics
unset key

set palette defined (0 "white", max_z "red")
set cbrange [0:max_z]
set view map

dir = "data/06.05.19/7"

set output sprintf("%s/%s_heatmap_%s.png", dir, quantity, n)

filename = sprintf("%s/flow/data_%s.out", dir, n)
splot filename using 1:2:3 with pm3d notitle
