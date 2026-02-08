#!/usr/bin/env python3
"""
Test data generator that mimics your process output.
Generates values similar to your example.
"""

import time
import random

def main():
    counter = 2282
    base_value = 0.33
    
    try:
        while True:
            # Generate a value with small random variations
            value = base_value + random.uniform(-0.3, 0.3)
            value = max(0.0, min(1.0, value))
            
            print(f"{counter} (60)-> {value}")
            counter += 1
            
            time.sleep(0.1)  # ~10 updates per second
            
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
