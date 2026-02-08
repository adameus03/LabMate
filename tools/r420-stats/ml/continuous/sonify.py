#!/usr/bin/env python3
"""
Real-time audio sonification with PROPER PACING.
Generates audio at exactly the rate it's consumed (no queue overflow, no dropped chunks).

Usage: python -u your_process.py | python sonify_final.py | pacat --format=s16
"""

import sys
import struct
import math
import threading
import queue
import time

# Audio parameters
SAMPLE_RATE = 44100
CHANNELS = 1

# Frequency mapping
BASE_FREQ = 200
FREQ_RANGE = 600

# Smoothing
SMOOTHING = 0.85

# Chunk size - this determines latency
CHUNK_SIZE = 2048  # ~46ms per chunk


class ToneGenerator:
    def __init__(self):
        self.current_freq = BASE_FREQ + FREQ_RANGE * 0.5
        self.target_freq = self.current_freq
        self.phase = 0.0
        self.update_count = 0
    
    def update_target(self, value):
        """Update target frequency from input value (0.0-1.0)"""
        self.target_freq = BASE_FREQ + value * FREQ_RANGE
        self.update_count += 1
    
    def generate_chunk(self, num_samples):
        """Generate a chunk of audio samples"""
        samples = []
        
        for _ in range(num_samples):
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
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    print(f"[{elapsed():.3f}s] Paced audio generator starting", file=sys.stderr, flush=True)
    print(f"[{elapsed():.3f}s] Chunk: {CHUNK_SIZE} samples = {CHUNK_SIZE/SAMPLE_RATE*1000:.1f}ms", 
          file=sys.stderr, flush=True)
    
    generator = ToneGenerator()
    input_queue = queue.Queue()
    
    def read_input():
        """Read input lines in separate thread"""
        try:
            buffer = b''
            while True:
                byte = sys.stdin.buffer.read(1)
                if not byte:
                    break
                if byte == b'\n':
                    try:
                        line = buffer.decode('utf-8')
                        arrival_time = time.time() - start_time
                        input_queue.put((line, arrival_time))
                        buffer = b''
                    except:
                        buffer = b''
                else:
                    buffer += byte
        except:
            pass
    
    # Start input reader thread
    reader = threading.Thread(target=read_input, daemon=True)
    reader.start()
    
    print(f"[{elapsed():.3f}s] Waiting for first input...", file=sys.stderr, flush=True)
    
    stdout_binary = sys.stdout.buffer
    started = False
    chunk_count = 0
    last_input_time = None
    
    # Calculate exact timing for audio chunks
    chunk_duration = CHUNK_SIZE / SAMPLE_RATE  # seconds per chunk
    next_chunk_time = None
    
    try:
        while True:
            # Process ALL available input immediately
            inputs_this_cycle = 0
            while not input_queue.empty():
                try:
                    line, arrival_time = input_queue.get_nowait()
                    process_time = elapsed()
                    
                    value = parse_value(line)
                    if value is not None:
                        generator.update_target(value)
                        inputs_this_cycle += 1
                        #freq = BASE_FREQ + value * FREQ_RANGE
                        #freq = BASE_FREQ + (value - 1) * 100
                        freq = BASE_FREQ * pow(2, 0.5*(value-1))
                        
                        if not started:
                            started = True
                            next_chunk_time = time.time()
                            print(f"[{process_time:.3f}s] FIRST INPUT: value={value:.4f} freq={freq:.0f}Hz | line: {line[:50]}", 
                                  file=sys.stderr, flush=True)
                        else:
                            time_since_last = arrival_time - last_input_time if last_input_time else 0
                            processing_delay = process_time - arrival_time
                            print(f"[{process_time:.3f}s] INPUT #{generator.update_count}: value={value:.4f} freq={freq:.0f}Hz | "
                                  f"delay={processing_delay*1000:.1f}ms gap={time_since_last*1000:.0f}ms | line: {line[:50]}", 
                                  file=sys.stderr, flush=True)
                        
                        last_input_time = arrival_time
                except queue.Empty:
                    break
            
            # Only generate audio after first input
            if started and next_chunk_time is not None:
                # Wait until it's time for the next chunk
                now = time.time()
                sleep_time = next_chunk_time - now
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Generate and write audio
                audio_data = generator.generate_chunk(CHUNK_SIZE)
                stdout_binary.write(audio_data)
                stdout_binary.flush()
                
                chunk_count += 1
                
                # Log every 50 chunks or when we got input
                if chunk_count % 50 == 0 or inputs_this_cycle > 0:
                    print(f"[{elapsed():.3f}s] Chunk #{chunk_count}: "
                          f"inputs={inputs_this_cycle} | "
                          f"freq={generator.current_freq:.0f}Hz->target={generator.target_freq:.0f}Hz",
                          file=sys.stderr, flush=True)
                
                # Schedule next chunk
                next_chunk_time += chunk_duration
                
                # If we've fallen behind, reset to now
                if next_chunk_time < time.time():
                    next_chunk_time = time.time()
            else:
                time.sleep(0.001)
    
    except KeyboardInterrupt:
        print(f"\n[{elapsed():.3f}s] Stopped. Total inputs: {generator.update_count}, chunks: {chunk_count}", 
              file=sys.stderr, flush=True)
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
