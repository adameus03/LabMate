#ifndef FH_H
#define FH_H

#include <stdint.h>

typedef struct freq_hop_table_entry {
  uint16_t channel_index;
  uint32_t frequency_khz;
} freq_hop_table_entry_t;

/**
 * @param channel_index Channel index (1-based)
 * @return Frequency in kHz
 */
uint32_t get_channel_frequency(uint16_t channel_index);

void fh_init();
int fh_check_all_channels_mapped();
void fh_create_entry(freq_hop_table_entry_t entry);

/**
 * Convert logical channel index to physical channel index. Logical channels are most often unordered, while the aim of this function is to provide ordered physical channels.
 * @param channel_index Logical channel index (1-based)
 * @return Physical channel index (0-based)
 */
uint16_t fh_get_physical_channel_index(uint16_t channel_index);

/**
 * Convert physical channel index to logical channel index.
 * @param physical_channel_index Physical channel index (0-based)
 * @return Logical channel index (1-based)
 */
uint16_t fh_get_logical_channel_index(uint16_t physical_channel_index);

#endif // FH_H