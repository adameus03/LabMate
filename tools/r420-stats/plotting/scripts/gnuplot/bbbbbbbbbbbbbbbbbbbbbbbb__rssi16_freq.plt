set xrange [902000:928000]
#set yrange [0:4096]
#set yrange [-32768:32767]
plot "../../../output/bbbbbbbbbbbbbbbbbbbbbbbb__rssi16_freq.dat" using 1:2
pause 1
reread
