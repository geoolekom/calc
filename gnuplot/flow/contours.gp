set term gif animate delay 10 size 1000, 500
set xrange [-0.5:10.5]
set yrange [-0.5:5.5]
unset surface

set style textbox opaque noborder margins 0.5, 0.5
set cntrparam bspline
set cntrparam levels auto 10
set cntrlabel font ",10"
set contour base
set contour
set view map

count = 400
quantity = ARG1
set output sprintf("data/%s_contours.gif", quantity)

do for [n = 0 : count - 1] {
    filename = sprintf("data/flow/%s_%03d.out", quantity, n)
    splot filename with lines  # , filename with labels boxed notitle
}
