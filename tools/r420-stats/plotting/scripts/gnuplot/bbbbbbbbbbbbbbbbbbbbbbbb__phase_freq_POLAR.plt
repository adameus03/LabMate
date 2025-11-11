# Enable polar coordinates
set polar
set angles degrees  # angles in degrees

# Set radius range (frequency values)
set rrange [900000:930000]

# Optional: grid
set grid polar

# Plot: angle is phase, radius is frequency
plot "../../../output/bbbbbbbbbbbbbbbbbbbbbbbb__phase_freq.dat" using ($2*360/4096):1 with points pt 7

pause 1
reread

