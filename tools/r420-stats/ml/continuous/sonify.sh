#!/bin/bash
# Real-time audio sonification using SoX
# Simple bash implementation for maximum compatibility

BASE_FREQ=200
FREQ_RANGE=600

echo "Starting real-time audio sonification (bash+SoX)..." >&2
echo "Frequency range: ${BASE_FREQ}Hz - $((BASE_FREQ + FREQ_RANGE))Hz" >&2
echo "Reading from stdin... (Ctrl+C to stop)" >&2

# Cleanup function
cleanup() {
    echo -e "\nStopping..." >&2
    # Kill any remaining play processes
    pkill -P $$ play 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Read input line by line
while IFS= read -r line; do
    # Extract the last number from the line
    value=$(echo "$line" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    
    if [ -n "$value" ]; then
        # Ensure value is between 0 and 1
        value=$(echo "$value" | awk '{if ($1 > 1) print 1; else if ($1 < 0) print 0; else print $1}')
        
        # Calculate frequency
        freq=$(echo "$BASE_FREQ + $value * $FREQ_RANGE" | bc -l)
        
        # Kill previous tone
        pkill -P $$ play 2>/dev/null
        
        # Play new tone in background with short duration and fade
        play -n -q synth 0.15 sine "$freq" fade h 0.01 0.15 0.04 &>/dev/null &
        
        # Print status
        printf "Value: %.4f -> Freq: %.1fHz\r" "$value" "$freq" >&2
    fi
done

cleanup
