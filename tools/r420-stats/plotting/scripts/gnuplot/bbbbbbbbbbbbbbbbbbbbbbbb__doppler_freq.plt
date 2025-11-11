set xrange [902000:928000]
set yrange [-32768:32767]
plot "../../../output/bbbbbbbbbbbbbbbbbbbbbbbb__doppler_freq.dat" using 1:2
pause 1
reread
