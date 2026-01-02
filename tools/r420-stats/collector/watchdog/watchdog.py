#!/usr/bin/env python3
import sys

emoji_line_counter = 0
data_lines = []
is_reading_data_lines = False
is_reading_pattern = False
def handle_line(line: str):
  global emoji_line_counter
  global data_lines
  global is_reading_data_lines
  global is_reading_pattern

  print(line)

  has_red   = "🟥" in line
  has_green = "🟩" in line
  has_blue  = "🟦" in line

  if is_reading_data_lines:
    if has_red or has_green or has_blue:
      print("ERROR: too little data lines")
      for i, data_line in enumerate(data_lines):
        print(f"(line {i+1}) {data_line}")
      exit(0)
    else:
      data_lines.append(line)
      if len(data_lines) == 43:
        is_reading_data_lines = False

  else:
    if has_red or has_green or has_blue:
      emoji_line_counter += 1
      if emoji_line_counter == 2:
        is_reading_data_lines = True        
        is_reading_pattern = True
        data_lines = []
        emoji_line_counter = 0
    elif is_reading_pattern:
      print("ERROR: too much data lines")
      for i, data_line in enumerate(data_lines):
        print(f"(line {i+1}) {data_line}")
      print(f"(line 44) {line}")
      exit(0)

def main():
  for raw in sys.stdin:
    handle_line(raw.rstrip("\n"))

if __name__ == "__main__":
    main()
