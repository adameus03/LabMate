set xrange [902000:928000]
set yrange [0:4096]

a(x) = (-0.291595*0.001*x + 271.344445)/(2*pi)*4096
wrap2048(x) = a(x) - 2048 * floor(a(x)/2048)

plot "../../../output/bbbbbbbbbbbbbbbbbbbbbbbb__phase_freq.dat" using 1:2 title "data", \
     wrap2048(x) title "fit mod 2048" with lines lc rgb "red", \
     (2048 + wrap2048(x)) title "fit mod 2048 + 2048" with lines lc rgb "blue"

pause 1
reread

