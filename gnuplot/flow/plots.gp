set term pngcairo size 1500, 800
set grid xtics lc rgb "#bbbbbb" lw 1 lt 0
set grid ytics lc rgb "#bbbbbb" lw 1 lt 0

data_dir = 'data/flow'
output_dir = 'data'
n = 600

# Поток через щель

set xlabel "t"
set ylabel "Ф"

quantity = 'flow'
set output sprintf("%s/%s.png", output_dir, quantity)
filename = sprintf("%s/%s.out", data_dir, quantity)
plot filename title ""

# Полуширина струи

set xlabel "x"
set ylabel "R"

quantity = 'radius'
set output sprintf("%s/%s.png", output_dir, quantity)
filename = sprintf("%s/%s_%03d.out", data_dir, quantity, n)
plot filename title ""

# Число Маха

set xlabel "x"
set ylabel "M"

quantity = 'mach_0'
set output sprintf("%s/%s.png", output_dir, quantity)
filename = sprintf("%s/%s_%03d.out", data_dir, quantity, n)
plot filename title ""

# Контуры

unset surface
set view map
set key outside
set style textbox opaque noborder margins 0.05, 0.05
set cntrlabel font "Times New Roman,10"

set contour base
set contour
set cntrparam bspline

# Контуры плотности, температуры и давления

set xtics 2
set ytics 2
set xlabel "X"
set ylabel "Y"

quantity = 'density'
set xrange [25:75]
set yrange [0:25]
set cntrparam levels incremental 0,0.01,0.2
set output sprintf("%s/%s_contours.png", output_dir, quantity)
filename = sprintf("%s/data_%03d.out", data_dir, n)
splot filename using 1:2:3 with lines title ""

quantity = 'temperature'
set xrange [25:75]
set yrange [0:25]
set cntrparam levels incremental 0,0.05,1.5
set output sprintf("%s/%s_contours.png", output_dir, quantity)
filename = sprintf("%s/data_%03d.out", data_dir, n)
splot filename using 1:2:4 with lines title ""

quantity = 'pressure'
set xrange [25:75]
set yrange [0:25]
set cntrparam levels incremental 0,0.01,0.3
set output sprintf("%s/%s_contours.png", output_dir, quantity)
filename = sprintf("%s/data_%03d.out", data_dir, n)
splot filename using 1:2:($3 * $4) with lines title ""

# Профили функции распределения

set term pngcairo size 1000, 750

set xtics 1
set ytics 1
set xlabel "v_x"
set ylabel "v_y"
set xrange [-0.5:3]
set yrange [-1.5:1.5]
set xtics 0.5
set ytics 0.5

quantity = 'function_screen_0'
set cntrparam levels auto 20
set output sprintf("%s/%s_contours.png", output_dir, quantity)
filename = sprintf("%s/%s_%03d.out", data_dir, quantity, n)
splot filename with lines title ""

quantity = 'function_screen_10_0'
set cntrparam levels auto 20
set output sprintf("%s/%s_contours.png", output_dir, quantity)
filename = sprintf("%s/%s_%03d.out", data_dir, quantity, n)
splot filename with lines title ""
