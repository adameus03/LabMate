#include "fh.h"
#include <assert.h>
#include <stdlib.h>

#define TOTAL_CHANNELS 50
uint32_t freq_hop_table[TOTAL_CHANNELS] = {0};
uint8_t channel_mapped[TOTAL_CHANNELS] = {0};
uint16_t channels_ranks_by_freq[TOTAL_CHANNELS] = {0}; // For converting logical to physical channel index
uint16_t logical_channels_sorted_by_freq[TOTAL_CHANNELS] = {0}; // For converting physical to locgical channel index
int channel_ranks_up_to_date = 0; // used to validate/invalidate the ranks array
int logical_channels_sorted_by_freq_up_to_date = 0; //used to validate/invalidate the corresponding sorted array

uint32_t get_channel_frequency(uint16_t channel_index) {
  assert(channel_index >= 1);
  assert(channel_index < TOTAL_CHANNELS + 1);
  return freq_hop_table[channel_index - 1];
}
void fh_init() {
  for (int i = 0; i < TOTAL_CHANNELS; i++) {
    channel_mapped[i] = 0;
  }
}
int fh_check_all_channels_mapped() {
  for (int i = 0; i < TOTAL_CHANNELS; i++) {
    if (channel_mapped[i] == 0) {
      return 0;
    }
  }
  return 1;
}
void fh_create_entry(freq_hop_table_entry_t entry) {
  assert(entry.channel_index >= 0 && entry.channel_index < TOTAL_CHANNELS);
  freq_hop_table[entry.channel_index] = entry.frequency_khz;
  channel_mapped[entry.channel_index] = 1;
  channel_ranks_up_to_date = 0; // Invalidate ranks array
}

static int compare_channel_frequencies(const void *indexA, const void *indexB) {
  uint16_t idxA = *(const uint16_t *)indexA;
  uint16_t idxB = *(const uint16_t *)indexB;
  uint32_t freqA = freq_hop_table[idxA - 1];
  uint32_t freqB = freq_hop_table[idxB - 1];
  if (freqA < freqB) return -1;
  if (freqA > freqB) return 1;
  return 0;
}

uint16_t fh_get_physical_channel_index(uint16_t channel_index) {
  assert(channel_index >= 1);
  assert(channel_index < TOTAL_CHANNELS + 1);
  if (!channel_ranks_up_to_date) {
    uint16_t temp[TOTAL_CHANNELS];
    for (int i = 0; i < TOTAL_CHANNELS; i++) {
      temp[i] = i + 1; // logical channel indices (1-based)
    }
    qsort(temp, TOTAL_CHANNELS, sizeof(uint16_t), compare_channel_frequencies);
    for (int i = 0; i < TOTAL_CHANNELS; i++) {
      channels_ranks_by_freq[temp[i] - 1] = i; // physical channel index (0-based)
    }
    channel_ranks_up_to_date = 1; // Ranks array is now valid
  }
  return channels_ranks_by_freq[channel_index - 1];
}

uint16_t fh_get_logical_channel_index(uint16_t physical_channel_index) {
  assert(physical_channel_index < TOTAL_CHANNELS);
  if (!logical_channels_sorted_by_freq_up_to_date) {
    for (int i = 0; i < TOTAL_CHANNELS; i++) {
      logical_channels_sorted_by_freq[i] = i + 1;
    }
    qsort(logical_channels_sorted_by_freq, TOTAL_CHANNELS, sizeof(uint16_t), compare_channel_frequencies);
    logical_channels_sorted_by_freq_up_to_date = 1;
  }
  return logical_channels_sorted_by_freq[physical_channel_index];
}