#!/usr/bin/env python3
"""
Real-time audio sonification using FFmpeg.
FFmpeg is widely available and has excellent audio compatibility.
"""

import sys
import subprocess
import struct
import math

# Audio parameters
SAMPLE_RATE = 44100
CHANNELS = 1

# Frequency parameters
BASE_FREQ = 200
FREQ_RANGE = 600

# Smoothing parameter
SMOOTHING = 0.90
UPDATE_INTERVAL = 0.1  # 10 Hz update rate


class FFmpegAudioPlayer:
    def __init__(self):
        self.current_freq = BASE_FREQ + FREQ_RANGE * 0.5
        self.target_freq = self.current_freq
        self.phase = 0.0
        self.samples_per_update = int(SAMPLE_RATE * UPDATE_INTERVAL)
        
        # Start FFmpeg process to play raw audio
        self.ffmpeg = subprocess.Popen(
            [
                'ffmpeg',
                '-f', 's16le',           # 16-bit signed little-endian
                '-ar', str(SAMPLE_RATE), # Sample rate
                '-ac', str(CHANNELS),    # Number of channels
                '-i', 'pipe:0',          # Input from stdin
                '-f', 'alsa',            # Output format (ALSA for Linux)
                'default',               # Default audio device
                '-loglevel', 'quiet'     # Suppress FFmpeg output
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
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
            
            # Convert to 16-bit signed integer
            sample_int = int(sample * 32767)
            sample_int = max(-32768, min(32767, sample_int))  # Clamp
            samples.append(struct.pack('<h', sample_int))
        
        return b''.join(samples)
    
    def write_audio(self, data):
        """Write audio data to FFmpeg process"""
        try:
            self.ffmpeg.stdin.write(data)
            self.ffmpeg.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False
    
    def close(self):
        """Clean up FFmpeg process"""
        try:
            if self.ffmpeg.poll() is None:
                self.ffmpeg.stdin.close()
                self.ffmpeg.terminate()
                self.ffmpeg.wait(timeout=1)
        except:
            try:
                self.ffmpeg.kill()
            except:
                pass


def parse_value(line):
    """Extract numerical value from input line"""
    line = line.strip()
    if not line:
        return None
    
    # Try to extract the last number from the line
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
    print("Starting real-time audio sonification (FFmpeg)...", file=sys.stderr)
    print(f"Frequency range: {BASE_FREQ}Hz - {BASE_FREQ + FREQ_RANGE}Hz", file=sys.stderr)
    print("Reading from stdin... (Ctrl+C to stop)", file=sys.stderr)
    
    try:
        player = FFmpegAudioPlayer()
    except FileNotFoundError:
        print("\nError: FFmpeg not found. Please install it:", file=sys.stderr)
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg", file=sys.stderr)
        print("  Fedora: sudo dnf install ffmpeg", file=sys.stderr)
        print("  Arch: sudo pacman -S ffmpeg", file=sys.stderr)
        sys.exit(1)
    
    last_value = 0.5
    
    try:
        sys.stdin.reconfigure(line_buffering=True)
        
        import select
        
        while True:
            # Check for input with timeout
            ready, _, _ = select.select([sys.stdin], [], [], UPDATE_INTERVAL)
            
            if ready:
                line = sys.stdin.readline()
                if not line:  # EOF
                    break
                    
                value = parse_value(line)
                if value is not None:
                    last_value = value
                    player.update_target(value)
                    freq = BASE_FREQ + value * FREQ_RANGE
                    print(f"Value: {value:.4f} -> Freq: {freq:.1f}Hz", 
                          file=sys.stderr, end='\r')
            
            # Generate and play audio continuously
            audio_data = player.generate_samples()
            if not player.write_audio(audio_data):
                print("\nFFmpeg process terminated unexpectedly", file=sys.stderr)
                break
    
    except KeyboardInterrupt:
        print("\nStopping...", file=sys.stderr)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
    finally:
        player.close()
        print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
