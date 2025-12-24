#include <signal.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#define TOTAL_CHANNELS 50
uint32_t freq_hop_table[TOTAL_CHANNELS] = {0};
uint8_t channel_mapped[TOTAL_CHANNELS] = {0};
int fh_ready_flag = 0;

typedef struct {
  uint8_t epc[12];
  uint16_t antenna_id;
  uint8_t peak_rssi;
  uint16_t rf_phase_angle;
  int16_t rf_doppler_frequency;
  int16_t peak_rssi_16bit;
  uint64_t first_seen_timestamp_utc_microseconds;
  uint64_t last_seen_timestamp_utc_microseconds;
  uint16_t tag_seen_count;
  uint16_t channel_index;
} epc_stats_t;

typedef struct {
  uint16_t channel_index;
  uint32_t frequency_khz;
} freq_hop_table_entry_t;

/**
 * @param channel_index Channel index (1-based)
 * @return Frequency in kHz
 */
uint32_t get_channel_frequency(uint16_t channel_index) {
  assert(channel_index >= 1);
  assert(channel_index < TOTAL_CHANNELS + 1);
  return freq_hop_table[channel_index - 1];
}

epc_stats_t parse_r420_log_line(const char *log_line) {
  epc_stats_t stats;
  memset(&stats, 0, sizeof(stats));
  sscanf(log_line, "R420 Log: Tag EPC: %02X%02X%02X%02X%02X%02X%02X%02X%02X%02X%02X%02X, Antenna ID: %u, Peak RSSI: %d, RF Phase Angle: %d, RF Doppler Frequency: %d, 16-bit peak rssi: %d, First Seen Timestamp (UTC): %llu us, Last Seen Timestamp (UTC): %llu us, Tag Seen Count: %u, Channel Index: %u",
    &stats.epc[0], &stats.epc[1], &stats.epc[2], &stats.epc[3], &stats.epc[4], &stats.epc[5],
    &stats.epc[6], &stats.epc[7], &stats.epc[8], &stats.epc[9], &stats.epc[10], &stats.epc[11],
    &stats.antenna_id, &stats.peak_rssi, &stats.rf_phase_angle, &stats.rf_doppler_frequency,
    &stats.peak_rssi_16bit, &stats.first_seen_timestamp_utc_microseconds,
    &stats.last_seen_timestamp_utc_microseconds, &stats.tag_seen_count, &stats.channel_index);
  return stats;
}

freq_hop_table_entry_t parse_r420_log_line_freq_hop(const char *log_line) {
  freq_hop_table_entry_t entry;
  memset(&entry, 0, sizeof(entry));
  sscanf(log_line, "R420 Log: r420_process_get_reader_capabilities_response_msg: FrequencyHopTable Entry %d: Frequency = %u kHz",
    &entry.channel_index, &entry.frequency_khz);
  return entry;
}

/**
 * Check if the log line starts with "R420 Log: Tag EPC: "
 * @return 1 if it does, 0 otherwise.
 */
int filter_log_line_epc(const char *log_line) {
  const char *prefix = "R420 Log: Tag EPC: ";
  return strncmp(log_line, prefix, strlen(prefix)) == 0;
} 

int filter_log_line_freq_hopping_table(const char *log_line) {
  const char *prefix = "R420 Log: r420_process_get_reader_capabilities_response_msg: FrequencyHopTable Entry";
  return strncmp(log_line, prefix, strlen(prefix)) == 0;
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
}

static void signal_handler(int sig){
  if (sig == SIGINT) {
    printf("Caught SIGINT, exiting.\n");
  }
}

int main(void) {
  signal(SIGINT, signal_handler);
  setvbuf(stdout, NULL, _IOLBF, 0);  // line-buffered stdout
  setvbuf(stdin, NULL, _IOLBF, 0);  // line-buffered stdin
  fh_init();
  char line[512];
  while (fgets(line, sizeof(line), stdin)) {
    line[strcspn(line, "\n")] = 0; // Remove newline
    if (filter_log_line_freq_hopping_table(line)) {
      printf("FH Table entry | %s\n", line);
      freq_hop_table_entry_t entry = parse_r420_log_line_freq_hop(line);
      printf("Parsed FH Table entry | Channel Index: %u, Frequency: %u kHz\n",
        entry.channel_index, entry.frequency_khz);
      fh_create_entry(entry);
    } else if (filter_log_line_epc(line)) {
      if (fh_ready_flag == 0) {
        if (fh_check_all_channels_mapped()) {
          printf("Frequency Hop Table fully mapped. Ready to process EPC log lines.\n");
          fh_ready_flag = 1;
        } else {
          printf("Frequency Hop Table not fully mapped. Skipping EPC log line processing until we get the full frequency hopping table.\n");
          continue;
        }
      }
      epc_stats_t stats = parse_r420_log_line(line);
      static int counter = 0;
      //printf("Counter: %d, Channel Index: %u, Frequency: %u kHz\n", ++counter, stats.channel_index, get_channel_frequency(stats.channel_index));
      printf("Measurement entry | EPC: %02X%02X%02X%02X%02X%02X%02X%02X%02X%02X%02X%02X, Antenna ID: %u, Peak RSSI: %d, RF Phase Angle: %d, RF Doppler Frequency: %d, 16-bit peak rssi: %d, First Seen Timestamp (UTC): %llu us, Last Seen Timestamp (UTC): %llu us, Tag Seen Count: %u, Channel Index: %u\n",
        stats.epc[0], stats.epc[1], stats.epc[2], stats.epc[3], stats.epc[4], stats.epc[5],
        stats.epc[6], stats.epc[7], stats.epc[8], stats.epc[9], stats.epc[10], stats.epc[11],
        stats.antenna_id, stats.peak_rssi, stats.rf_phase_angle, stats.rf_doppler_frequency,
        stats.peak_rssi_16bit, (unsigned long long)stats.first_seen_timestamp_utc_microseconds,
        (unsigned long long)stats.last_seen_timestamp_utc_microseconds, stats.tag_seen_count, stats.channel_index);
    }
    
  }
  fprintf(stderr, "Input stream closed. Exiting.\n");
  return 0;
}