#ifndef MBUFFER_H
#define MBUFFER_H

#include <stdint.h>
#include <stdio.h>

#define NUM_REFERENCE_TAGS 12
//#define NUM_REFERENCE_TAGS 0
#define NUM_TRACKED_ASSETS 1
#define NUM_ANTENNAS 2
#define NUM_CHANNELS 50
#define NUM_MEASUREMENTS 2
#define MOVING_AVERAGE_WINDOW 5

//#define RSSI_MIN_VALUE -32768 // Minimum int16_t value
//#define RSSI_MAX_VALUE 32767  // Maximum int16_t value
#define RSSI_MIN_VALUE -10000
#define RSSI_MAX_VALUE -3000
#define DOPPLER_MIN_VALUE -12000
#define DOPPLER_MAX_VALUE 12000

typedef struct mbuffer_unit {
  int16_t rssi[NUM_CHANNELS][NUM_MEASUREMENTS];
  uint16_t phase_angle[NUM_CHANNELS][NUM_MEASUREMENTS];
  int16_t doppler_frequency[NUM_CHANNELS][NUM_MEASUREMENTS];
  uint32_t counter[NUM_CHANNELS];
} mbuffer_unit_t;

typedef struct mbuffer_normalized_unit {
  double rssi[NUM_CHANNELS][NUM_MEASUREMENTS];
  double phase_angle[NUM_CHANNELS][NUM_MEASUREMENTS];
  double doppler_frequency[NUM_CHANNELS][NUM_MEASUREMENTS];
  uint32_t counter[NUM_CHANNELS];
} mbuffer_normalized_unit_t;

typedef struct mbuffer { // measurement buffer struct
  mbuffer_unit_t data[NUM_ANTENNAS * (NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS)];
} mbuffer_t;

mbuffer_unit_t mbuffer_get_unit(mbuffer_t *mbuf, uint8_t antenna_id, uint32_t tag_index);

mbuffer_normalized_unit_t mbuffer_get_normalized_unit(mbuffer_t *mbuf, uint8_t antenna_id, uint32_t tag_index);

void mbuffer_flush(mbuffer_t *mbuf);

void mbuffer_update_unit(mbuffer_t *mbuf, uint8_t antenna_id, uint32_t tag_index, uint32_t channel,
                         int16_t rssi, uint16_t phase_angle, int16_t doppler_frequency);

// void mbuffer_normalized_unit_to_file(mbuffer_normalized_unit_t* normalized_unit, FILE* file);
#endif // MBUFFER_H