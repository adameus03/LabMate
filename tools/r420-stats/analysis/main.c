#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#ifndef __USE_MISC
  #define __USE_MISC
#endif
#include <math.h>

#define DATA_FILE_PATH "../output/bbbbbbbbbbbbbbbbbbbbbbbb__phase_freq.dat"

int main(void) {
    FILE *file = fopen(DATA_FILE_PATH, "r");
    if (file == NULL) {
        perror("Failed to open data file");
        return 1;
    }

    unsigned int frequency;
    int phase_angle;

    //printf("Frequency (kHz)\tPhase Angle (degrees)\n");
    //printf("-----------------------------------------\n");

    double sum_xx = 0.0;
    double sum_xy = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    int n = 0;

    while (fscanf(file, "%u\t%d", &frequency, &phase_angle) == 2) {
      //if (frequency > 907500 || phase_angle >= 2048) {
      if (frequency < 920000 || phase_angle >= 2048) {
        // Skip frequencies above 907.5 MHz
        continue;
      }

      //phase_angle %= 2048;
      assert(phase_angle >= 0 && phase_angle < 2048);

      printf("Valid data point: Frequency = %u kHz, Phase Angle = %d pseudodegrees\n", frequency, phase_angle);

      double _frequency = frequency / 1000.0; // convert to MHz
      double _phase_angle = phase_angle / 4096.0 * 2.0 * M_PI; // convert to radians

      sum_xx += _frequency * _frequency;
      sum_xy += _frequency * _phase_angle;
      sum_x += _frequency;
      sum_y += _phase_angle;
      n++;
    }

    fclose(file);

    if (n == 0) {
        printf("No data points found.\n");
        return 1;
    } 

    printf("Processed %d valid data points.\n", n);
    printf("Sum_xx: %llu, Sum_xy: %llu, Sum_x: %llu, Sum_y: %llu\n",
           (unsigned long long)sum_xx,
           (unsigned long long)sum_xy,
           (unsigned long long)sum_x,
           (unsigned long long)sum_y);

    
    double avg_x = (double)sum_x / n;
    double avg_y = (double)sum_y / n;
    double avg_xx = (double)sum_xx / n;
    double avg_xy = (double)sum_xy / n;
    printf("Average_x: %lf, Average_y: %lf, Average_xx: %lf, Average_xy: %lf\n",
           avg_x, avg_y, avg_xx, avg_xy);

    
    
    //double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    //double intercept = (sum_y - slope * sum_x) / n;

    double slope = (avg_xy - avg_x * avg_y) / (avg_xx - avg_x * avg_x);
    double intercept = avg_y - slope * avg_x;

    printf("Linear Regression Result:\n");
    printf("Slope: %lf radians/MHz\n", slope);
    printf("Intercept: %lf radians\n", intercept);
    printf("Equation: Phase Angle = %lf * Frequency + %lf\n", slope, intercept);
    printf("----------------------------------------\n");
    printf("Note: phase angle is in radians and frequency is in MHz.\n");

    return 0;
}