from pyo import *

# Start the audio server
s = Server().boot()
s.start()

# Create an initial oscillator
freq = Sig(200)  # Signal for frequency (modifiable in real-time)
osc = Sine(freq=freq, mul=0.5).out()  # Sine wave oscillator with volume control

# Function to update frequency
def update_frequency(new_freq):
    freq.value = new_freq  # Update the frequency in real-time

# Example: Simulate frequency changes
import time
try:
    while True:
        # Simulate receiving a frequency value
        hex_signal = input("Enter signal strength (hex): ")
        if hex_signal == "00":
            continue  # Skip silence
        decimal_value = int(hex_signal, 16)
        new_freq = 200 + decimal_value * 10  # Map to frequency range
        update_frequency(new_freq)
except KeyboardInterrupt:
    s.stop()
