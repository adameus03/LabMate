### Building
```
gcc main.c r420.c -o r420-stats
```

### Building the collector
```
cd collector
gcc main.c -o collector
```

### Usage
```
./r420-stats | collector/collector
```
View the realtime stats with 
```
cd plotting/scripts
gnuplot <script_name>.plt
```
To gracefully stop the r420-stats and collector processes, it's enough to send a SIGINT (Ctrl+C) using the tty. 

### Linear regression analysis
To perform linear regression analysis on the collected phase-freq data, use the analysis program.

While in the `analysis` directory, compile the analysis program:
```
gcc analysis.c -o analyse -lm
```
Run the analysis:
```
./analyse > analysis_result.log
```
