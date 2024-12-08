#!/bin/bash

TAG_DIR=$1; # for example /mnt/rscs/uhf0

# Function to convert hex signal strength to frequency
function hex_to_freq() {
  local hex_value=$1;
  # Convert hex to decimal
  local decimal_value=$((16#$hex_value));
  # Map decimal value to frequency range (e.g., 200 Hz to 2000 Hz)
  if [ "$decimal_value" -lt "176" ]; then
    REPLY=0;
  else
    #REPLY=$(awk "BEGIN {print 11.3924050633 * $decimal_value - 1905.06329114}");
    REPLY=$(awk "BEGIN {print 36.7088607595 * $decimal_value - 6360.75949367}");
  fi;
}

function loop() {
  while true; do
    echo -n '100000 26' > "$TAG_DIR"/driver/measure;
    local read_rate="$(cat "$TAG_DIR/read_rate")";
    local rssi="$(cat "$TAG_DIR/rssi")";
    hex_to_freq "$rssi";
    local freq="$REPLY";
    echo "$freq";
    #sox -n -q -t alsa default synth 0.05 sine "$freq" vol 0.05;
  done;
}

loop;