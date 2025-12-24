
set yrange [0.0 to 1.0]        # adjust to your expected data range
set grid
#set autoscale x
#set autoscale y

plot "../../../outputs/mbuf_12_1_doppler.csv" using ( $1 != 0.0 ? $1 : 1/0 ) title "RSSI"

# For real-time updates
pause 0.01               # 0.01 seconds
reread                   # re-run this script
