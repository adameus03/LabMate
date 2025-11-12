set xrange [902000:928000]
set yrange [0:4096]
plot "../../../output/bbbbbbbbbbbbbbbbbbbbbbbb__phase_freq.dat" using 1:2 title "data", \
     ((-0.283356*0.001*x + 257.584430)/(2*pi)*4096) title "fit" with lines lc rgb "red"
pause 1
reread

