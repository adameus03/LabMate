#!/usr/bin/env python3
import sys

def value_to_visualization(value):
    """Convert a float value to a visualization with 18 red squares and 1 purple square."""
    # Clamp value to [0, 1] range
    value = max(0.0, min(1.0, value))
    
    # Calculate position (0-18, where position 18 means the purple square is at the end)
    # The purple square represents the value's position across the range
    position = round(value * 18)
    
    # Build the visualization string
    squares = []
    for i in range(19):
        if i == position:
            squares.append('🟪')
        else:
            squares.append('🟥')
    
    return f"{''.join(squares)} {position + 1} ({value})"

def main():
    # Disable buffering on stdout
    sys.stdout.reconfigure(line_buffering=True)
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            value = float(line)
            visualization = value_to_visualization(value)
            print(visualization, flush=True)
        except ValueError:
            # Skip lines that aren't valid floats
            continue

if __name__ == '__main__':
    main()
