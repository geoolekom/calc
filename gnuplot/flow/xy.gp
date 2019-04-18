quantity = ARG1
count = ARG2
step = ARG3
if (!exists("step")) step = 1

set term gif animate delay 10 size 2000, 1000
set xrange [0:100]
set yrange [0:25]

set output sprintf("data/%s_xy.gif", quantity)

do for [n = 0 : count - 1 : step] {
    filename = sprintf("data/17.04.19/flow 6 no screen/%s_%03d.out", quantity, n)
    plot filename
}
