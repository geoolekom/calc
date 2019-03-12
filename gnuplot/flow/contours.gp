quantity = ARG1
count = ARG2
step = ARG3
if (!exists("step")) step = 1

set term gif animate delay 10 size 2000, 1000
set xrange [6:18]
set yrange [0:5]
unset surface

set style textbox opaque noborder margins 0.5, 0.5
set cntrparam bspline
set cntrparam levels incremental 0,0.02,0.2
set cntrlabel font ",10"
set contour base
set contour
set view map

set output sprintf("data/%s_contours.gif", quantity)

do for [n = 0 : count - 1 : step] {
    filename = sprintf("data/flow/%s_%03d.out", quantity, n)
    splot filename with lines  # , filename with labels boxed notitle
}
