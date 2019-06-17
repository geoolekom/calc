quantity = ARG1
count = ARG2
step = ARG3
if (!exists("step")) step = 1

set term gif animate delay 10 size 2000, 1000
set xrange [20:70]
set xtics 5
set yrange [0:25]
set ytics 5
unset surface

set grid ytics lc rgb "#bbbbbb" lw 1 lt 0
set grid xtics lc rgb "#bbbbbb" lw 1 lt 0

set cntrparam levels incremental 0,0.02,1
set style textbox opaque noborder margins 0.5, 0.5
set cntrparam bspline
set cntrlabel font ",10"
set contour base
set contour
set key outside
set view map

dir = "data/06.05.19/9"

set output sprintf("%s/%s_contours.gif", dir, quantity)

do for [n = 0 : count - 1 : step] {
    filename = sprintf("%s/flow/data_%03d.out", dir, n)
    splot filename using 1:2:3 with lines title ""
}
