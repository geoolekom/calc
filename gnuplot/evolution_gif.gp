set term gif animate delay 10 size 800, 800
set xrange [-0.5:10.5]
set yrange [-0.5:5.5]
set zrange [0:1.2]

set output "data/flow.gif"
count = 100

do for [n = 0 : count - 1] {
    filename = sprintf("data/flow/density_%03d.out", n)
    splot filename with lines
}
