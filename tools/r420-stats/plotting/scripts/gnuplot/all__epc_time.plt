set terminal qt size 1200,800 persist
set title "RFID EPC Waterfall"
set xlabel "EPC"
set ylabel "Time"
set ydata time
set timefmt "%s"
set format y "%H:%M:%S"
set xtics rotate by -45
set grid
set style data points
set pointsize 0.5

# This creates numbered columns for each unique EPC
plot "< awk 'NR==FNR{epc[$2]=++n; next} {print epc[$2], $1/1000}' ../../../output/all__epc_time.dat ../../../output/all__epc_time.dat" using 1:2 with points pt 5 notitle

pause 1
reread
