quantity = ARG1
count = ARG2
step = ARG3
if (!exists("step")) step = 1

set term gif animate delay 10 size 2000, 1000
set xrange [25:75]
set xtics -100,1,100
set yrange [0:25]
set ytics -100,1,100
unset surface

set grid ytics lc rgb "#bbbbbb" lw 1 lt 0
set grid xtics lc rgb "#bbbbbb" lw 1 lt 0

set cntrparam levels incremental 0,0.01,0.2
set style textbox opaque noborder margins 0.5, 0.5
set cntrparam bspline
set cntrlabel font ",10"
set contour base
set contour
set key outside
set view map

set output sprintf("data/%s_contours.gif", quantity)

do for [n = 0 : count - 1 : step] {
    filename = sprintf("data/flow/%s_%03d.out", quantity, n)
    splot filename with lines title ""
}
