#!/bin/bash

# Function to convert hex signal strength to frequency
hex_to_freq() {
    hex_value=$1
    # Convert hex to decimal
    decimal_value=$((16#$hex_value))
    # Map decimal value to frequency range (e.g., 200 Hz to 2000 Hz)
    echo $((200 + decimal_value * 10))
}

# Main loop
while read -r line; do
    # Extract the second column (signal strength in hex)
    signal_hex=$(echo "$line" | awk '{print $2}')

    # Skip if the value is "00" (no signal)
    if [[ $signal_hex == "00" ]]; then
        sox -n -q -t alsa default synth 0.1 sine 0 vol 0.01 # Short silence
        continue
    fi

    # Convert hex signal strength to frequency
    frequency=$(hex_to_freq "$signal_hex")

    # Generate sound for 0.1 second at the calculated frequency
    sox -n -q -t alsa default synth 0.1 sine "$frequency" vol 0.01;
done
~          