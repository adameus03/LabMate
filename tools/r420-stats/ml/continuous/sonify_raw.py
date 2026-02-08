#!/usr/bin/env python3
"""
Real-time audio sonification - generates raw s16le PCM audio to stdout.
Usage: your_process | python sonify_raw.py | pacat --format=s16le --rate=44100 --channels=1
"""

import sys
import struct
import math
import select

# Audio parameters
SAMPLE_RATE = 44100
CHANNELS = 1

# Frequency parameters
BASE_FREQ = 200
FREQ_RANGE = 600

# Smoothing parameter
SMOOTHING = 0.90
UPDATE_INTERVAL = 0.1  # 10 Hz update rate


class ToneGenerator:
    def __init__(self):
        self.current_freq = BASE_FREQ + FREQ_RANGE * 0.5
        self.target_freq = self.current_freq
        self.phase = 0.0
        self.samples_per_update = int(SAMPLE_RATE * UPDATE_INTERVAL)
    
    def update_target(self, value):
        """Update target frequency based on input value (0.0-1.0)"""
        self.target_freq = BASE_FREQ + value * FREQ_RANGE
    
    def generate_samples(self):
        """Generate audio samples with smooth frequency transition"""
        samples = []
        
        for _ in range(self.samples_per_update):
            # Smooth transition to target frequency
            self.current_freq = (SMOOTHING * self.current_freq + 
                               (1 - SMOOTHING) * self.target_freq)
            
            # Generate sine wave sample
            sample = 0.3 * math.sin(self.phase)
            self.phase += 2.0 * math.pi * self.current_freq / SAMPLE_RATE
            
            # Keep phase in reasonable range
            if self.phase > 2.0 * math.pi * 1000:
                self.phase -= 2.0 * math.pi * 1000
            
            # Convert to 16-bit signed integer (little-endian)
            sample_int = int(sample * 32767)
            sample_int = max(-32768, min(32767, sample_int))
            samples.append(struct.pack('<h', sample_int))
        
        return b''.join(samples)


def parse_value(line):
    """Extract numerical value from input line"""
    line = line.strip()
    if not line:
        return None
    
    parts = line.split()
    for part in reversed(parts):
        try:
            value = float(part)
            return max(0.0, min(1.0, value))
        except ValueError:
            continue
    
    try:
        value = float(line)
        return max(0.0, min(1.0, value))
    except ValueError:
        return None


def main():
    print("Tone generator started", file=sys.stderr)
    print(f"Frequency range: {BASE_FREQ}Hz - {BASE_FREQ + FREQ_RANGE}Hz", file=sys.stderr)
    
    generator = ToneGenerator()
    
    # Make stdout binary for raw audio output
    stdout_binary = sys.stdout.buffer
    
    try:
        sys.stdin.reconfigure(line_buffering=True)
        
        while True:
            # Check for input with timeout
            ready, _, _ = select.select([sys.stdin], [], [], UPDATE_INTERVAL)
            
            if ready:
                line = sys.stdin.readline()
                if not line:  # EOF
                    break
                    
                value = parse_value(line)
                if value is not None:
                    generator.update_target(value)
                    freq = BASE_FREQ + value * FREQ_RANGE
                    print(f"Value: {value:.4f} -> Freq: {freq:.1f}Hz", 
                          file=sys.stderr, end='\r')
            
            # Generate and output audio continuously
            audio_data = generator.generate_samples()
            stdout_binary.write(audio_data)
            stdout_binary.flush()
    
    except KeyboardInterrupt:
        print("\nStopped", file=sys.stderr)
    except BrokenPipeError:
        # Output pipe closed (pacat stopped)
        pass


if __name__ == "__main__":
    main()
