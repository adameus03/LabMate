#include "mbuffer.h"
#include <assert.h>
#include <stddef.h>
#include "fh.h"

mbuffer_unit_t mbuffer_get_unit(mbuffer_t *mbuf, uint8_t antenna_id, uint32_t tag_index) {
  assert(antenna_id >= 1);
  assert(antenna_id <= NUM_ANTENNAS);
  assert(tag_index < NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS);
  return mbuf->data[tag_index * NUM_ANTENNAS + antenna_id - 1];
}

mbuffer_normalized_unit_t mbuffer_get_normalized_unit(mbuffer_t *mbuf, uint8_t antenna_id, uint32_t tag_index) {
  assert(antenna_id <= NUM_ANTENNAS);
  assert(tag_index < NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS);
  mbuffer_unit_t unit = mbuffer_get_unit(mbuf, antenna_id, tag_index);
  mbuffer_normalized_unit_t normalized_unit = {0};
  for (size_t i = 0; i < NUM_CHANNELS; i++) {
    for (size_t j = 0; j < NUM_MEASUREMENTS; j++) {
      // scale rssi from -32768..32767 to 0.0..1.0
      normalized_unit.rssi[i][j] = (double)(unit.rssi[i][j] - RSSI_MIN_VALUE) / (RSSI_MAX_VALUE - RSSI_MIN_VALUE);
      // scale phase_angle from 0..4095 to 0.0..1.0
      normalized_unit.phase_angle[i][j] = ((double)(unit.phase_angle[i][j])) / 4095.0;
      // scale doppler_frequency from -32768..32767 to 0.0..1.0
      normalized_unit.doppler_frequency[i][j] = ((double)(unit.doppler_frequency[i][j] - DOPPLER_MIN_VALUE)) / (double)(DOPPLER_MAX_VALUE - DOPPLER_MIN_VALUE);
    
      // Switch to -1.0..1.0
      normalized_unit.rssi[i][j] = normalized_unit.rssi[i][j] * 2.0 - 1.0;
      normalized_unit.phase_angle[i][j] = normalized_unit.phase_angle[i][j] * 2.0 - 1.0;
      normalized_unit.doppler_frequency[i][j] = normalized_unit.doppler_frequency[i][j] * 2.0 - 1.0;
    }
  }
  return normalized_unit;
}

void mbuffer_flush(mbuffer_t *mbuf) {
  assert(mbuf != NULL);
  assert(RSSI_MIN_VALUE < 0);
  for (size_t i = 0; i < NUM_ANTENNAS * (NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS); i++) {
    for (size_t j = 0; j < NUM_CHANNELS; j++) {
      for (size_t k = 0; k < NUM_MEASUREMENTS; k++) {
        mbuf->data[i].rssi[j][k] = RSSI_MIN_VALUE; // Minimum int16_t value
        mbuf->data[i].phase_angle[j][k] = 0;
        mbuf->data[i].doppler_frequency[j][k] = 0;
      }
      mbuf->data[i].counter[j] = 0;
    }
  }
}

void mbuffer_update_unit(mbuffer_t *mbuf, uint8_t antenna_id, uint32_t tag_index, uint32_t channel,
                         int16_t rssi, uint16_t phase_angle, int16_t doppler_frequency) {
  
  assert(antenna_id >= 1);
  assert(antenna_id <= NUM_ANTENNAS);
  assert(tag_index < NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS);
  assert(channel <= NUM_CHANNELS);
  assert(channel >= 1);
  uint16_t physical_channel = fh_get_physical_channel_index(channel);
  assert(physical_channel < NUM_CHANNELS);
  mbuffer_unit_t *unit = &mbuf->data[tag_index * NUM_ANTENNAS + antenna_id - 1];
  uint32_t* counter = &unit->counter[physical_channel];
  // if (*counter <= MOVING_AVERAGE_WINDOW) { // Use averaging filter
  //   if (*counter == 0) {
  //     unit->rssi[physical_channel] = rssi;
  //     unit->phase_angle[physical_channel] = phase_angle;
  //     unit->doppler_frequency[physical_channel] = doppler_frequency;
  //   } else {
  //     double alpha = 1.0 - (1.0 / (double)(*counter + 1));
  //     unit->rssi[physical_channel] = (int16_t)(alpha * unit->rssi[physical_channel] + (1.0 - alpha) * rssi);
  //     unit->phase_angle[physical_channel] = (uint16_t)(alpha * unit->phase_angle[physical_channel] + (1.0 - alpha) * phase_angle);
  //     unit->doppler_frequency[physical_channel] = (uint16_t)(alpha * unit->doppler_frequency[physical_channel] + (1.0 - alpha) * doppler_frequency);
  //   }
    
  // } else { // Use moving average with MOVING_AVERAGE_WINDOW
  //   double alpha = (double)*counter / (double)(*counter + 1);
  //   unit->rssi[physical_channel] = (int16_t)(alpha * unit->rssi[physical_channel] + (1.0 - alpha) * rssi);
  //   unit->phase_angle[physical_channel] = (uint16_t)(alpha * unit->phase_angle[physical_channel] + (1.0 - alpha) * phase_angle);
  //   unit->doppler_frequency[physical_channel] = (uint16_t)(alpha * unit->doppler_frequency[physical_channel] + (1.0 - alpha) * doppler_frequency);
  // }
  unit->rssi[physical_channel][*counter % NUM_MEASUREMENTS] = rssi;
  unit->phase_angle[physical_channel][*counter % NUM_MEASUREMENTS] = phase_angle;
  unit->doppler_frequency[physical_channel][*counter % NUM_MEASUREMENTS] = doppler_frequency;
  (*counter)++;
}

// void mbuffer_normalized_unit_to_file(mbuffer_normalized_unit_t* normalized_unit, FILE* file) {
//   assert(normalized_unit != NULL);
//   assert(file != NULL);
//   for (size_t i = 0; i < NUM_CHANNELS; i++) {
//     fprintf(file, "%f,%f,%f\n",
//             normalized_unit->rssi[i],
//             normalized_unit->phase_angle[i],
//             normalized_unit->doppler_frequency[i]);
//   }
// }