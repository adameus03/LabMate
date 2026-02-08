#!/usr/bin/env python3
"""
Debug version - shows exactly what's happening
"""

import sys
import struct
import math
import select

SAMPLE_RATE = 44100
BASE_FREQ = 200
FREQ_RANGE = 600
SMOOTHING = 0.90
UPDATE_INTERVAL = 0.1

class ToneGenerator:
    def __init__(self):
        self.current_freq = BASE_FREQ + FREQ_RANGE * 0.5
        self.target_freq = self.current_freq
        self.phase = 0.0
        self.samples_per_update = int(SAMPLE_RATE * UPDATE_INTERVAL)
        self.update_count = 0
    
    def update_target(self, value):
        self.target_freq = BASE_FREQ + value * FREQ_RANGE
        self.update_count += 1
    
    def generate_samples(self):
        samples = []
        for _ in range(self.samples_per_update):
            self.current_freq = (SMOOTHING * self.current_freq + 
                               (1 - SMOOTHING) * self.target_freq)
            sample = 0.3 * math.sin(self.phase)
            self.phase += 2.0 * math.pi * self.current_freq / SAMPLE_RATE
            if self.phase > 2.0 * math.pi * 1000:
                self.phase -= 2.0 * math.pi * 1000
            sample_int = int(sample * 32767)
            sample_int = max(-32768, min(32767, sample_int))
            samples.append(struct.pack('<h', sample_int))
        return b''.join(samples)

def parse_value(line):
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
    print("DEBUG MODE - watching input carefully", file=sys.stderr)
    
    generator = ToneGenerator()
    stdout_binary = sys.stdout.buffer
    
    # Try reading without select first
    sys.stdin.reconfigure(line_buffering=True)
    
    import threading
    import queue
    
    input_queue = queue.Queue()
    
    def read_input():
        """Read input in a separate thread"""
        try:
            for line in sys.stdin:
                input_queue.put(line)
        except:
            pass
    
    # Start input reader thread
    reader_thread = threading.Thread(target=read_input, daemon=True)
    reader_thread.start()
    
    try:
        iteration = 0
        while True:
            # Check for new input
            got_input = False
            while not input_queue.empty():
                try:
                    line = input_queue.get_nowait()
                    value = parse_value(line)
                    if value is not None:
                        generator.update_target(value)
                        got_input = True
                        print(f"[{iteration}] INPUT! value={value:.4f} target={generator.target_freq:.1f}Hz current={generator.current_freq:.1f}Hz updates={generator.update_count}", 
                              file=sys.stderr)
                except queue.Empty:
                    break
            
            # Generate audio
            audio_data = generator.generate_samples()
            stdout_binary.write(audio_data)
            stdout_binary.flush()
            
            iteration += 1
            if iteration % 10 == 0 and not got_input:
                print(f"[{iteration}] No new input. current={generator.current_freq:.1f}Hz target={generator.target_freq:.1f}Hz", 
                      file=sys.stderr)
    
    except KeyboardInterrupt:
        print("\nStopped", file=sys.stderr)
    except BrokenPipeError:
        pass
    
    print(f"Total updates received: {generator.update_count}", file=sys.stderr)

if __name__ == "__main__":
    main()
