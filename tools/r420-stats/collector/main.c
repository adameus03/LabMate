#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

/*
  Relevant log lines begin with "R420 Log: "
  Example log line:
  R420 Log: Tag EPC: 30395DFA82F8960000291BE9, Antenna ID: 1, Peak RSSI: 196, RF Phase Angle: 804, RF Doppler Frequency: -273, 16-bit peak rssi: -6050, First Seen Timestamp (UTC): 1762871268217261 us, Last Seen Timestamp (UTC): 1762871268217261 us, Tag Seen Count: 1, Channel Index: 10
*/

#define TOTAL_CHANNELS 50
uint32_t freq_hop_table[TOTAL_CHANNELS] = {
  903250, 915750, 917250, 920250, 923750,
  921750, 907750, 920750, 906250, 912250,
  908750, 908250, 909750, 922250, 927250,
  913750, 924250, 911250, 926750, 913250,
  918250, 905250, 915250, 903750, 911750,
  914250, 905750, 902750, 918750, 909250,
  904750, 926250, 904250, 906750, 916750,
  925750, 919750, 922750, 921250, 919250,
  916250, 910750, 923250, 907250, 924750,
  914750, 910250, 917750, 925250, 912750
};
uint8_t channel_mapped[TOTAL_CHANNELS] = {0};

/**
 * @param channel_index Channel index (1-based)
 * @return Frequency in kHz
 */
uint32_t get_channel_frequency(uint16_t channel_index) {
  assert(channel_index >= 1);
  assert(channel_index < TOTAL_CHANNELS + 1);
  return freq_hop_table[channel_index - 1];
}

const char* output_paths[] = {
  "./output/bbbbbbbbbbbbbbbbbbbbbbbb__rssi8_time.dat",
  "./output/bbbbbbbbbbbbbbbbbbbbbbbb__rssi16_time.dat",
  "./output/bbbbbbbbbbbbbbbbbbbbbbbb__phase_freq.dat",
  "./output/bbbbbbbbbbbbbbbbbbbbbbbb__doppler_freq.dat",
  "./output/bbbbbbbbbbbbbbbbbbbbbbbb__phase_time.dat",
  "./output/bbbbbbbbbbbbbbbbbbbbbbbb__rssi16_freq.dat",
  "./output/bbbbbbbbbbbbbbbbbbbbbbbb__ccp_time.dat", // channel-corrected phase vs time
};

#define NUM_OUTPUT_FILES 7
FILE* output_files[NUM_OUTPUT_FILES] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL };

static void signal_handler(int sig){
  if (sig == SIGINT) {
    printf("Caught SIGINT, closing output files...\n");
    for (int i = 0; i < NUM_OUTPUT_FILES; i++) {
      if (output_files[i]) {
        fclose(output_files[i]);
        output_files[i] = NULL;
      }
    }
  }
}

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
 * Make sure the ./output directory is freshly created.
 */
void prepare_output_dir() {
  const char *dir_path = "./output";
  struct stat st = {0};
  if (stat(dir_path, &st) == -1) {
    assert(0 == mkdir(dir_path, 0700));
  } else {
    for (int i = 0; i < NUM_OUTPUT_FILES; i++) {
      // delete output_paths[i]
      printf("Removing existing output file: %s\n", output_paths[i]);
      int rv = remove(output_paths[i]);
      if (rv != 0 && errno != ENOENT) {
        fprintf(stderr, "Failed to remove existing output file %s, errno: %d\n", output_paths[i], errno);
        assert(0);
      }
    }
    //int rv = rmdir(dir_path);
    int rv = remove(dir_path);
    if (rv != 0) {
      fprintf(stderr, "Failed to remove existing output directory, errno: %d\n", errno);
      assert(0);
    }
    assert(0 == mkdir(dir_path, 0700));
  }   
}

void handle_bbbbbbbbbbbbbbbbbbbbbbbb_rssi8_time(const epc_stats_t* stats) {
  for (int i = 0; i < 12; i++) {
    if (stats->epc[i] != 0xBB) {
      return;
    }
  }

  if (output_files[0] == NULL) {
    output_files[0] = fopen(output_paths[0], "a");
    if (output_files[0] == NULL) {
      perror("Failed to open output file for RSSI 8-bit vs Time");
      return;
    }
    setvbuf(output_files[0], NULL, _IOLBF, 0); // line-buffered
  }
  fprintf(output_files[0], "%llu\t%d\n",
          (unsigned long long)stats->last_seen_timestamp_utc_microseconds,
          stats->peak_rssi);
}

void handle_bbbbbbbbbbbbbbbbbbbbbbbb_rssi16_time(const epc_stats_t* stats) {
  for (int i = 0; i < 12; i++) {
    if (stats->epc[i] != 0xBB) {
      return;
    }
  }

  if (output_files[1] == NULL) {
    output_files[1] = fopen(output_paths[1], "a");
    if (output_files[1] == NULL) {
      perror("Failed to open output file for RSSI 16-bit vs Time");
      return;
    }
    setvbuf(output_files[1], NULL, _IOLBF, 0); // line-buffered
  }
  fprintf(output_files[1], "%llu\t%d\n",
          (unsigned long long)stats->last_seen_timestamp_utc_microseconds,
          stats->peak_rssi_16bit);
}

void handle_bbbbbbbbbbbbbbbbbbbbbbbb_phase_freq(const epc_stats_t* stats) {
  for (int i = 0; i < 12; i++) {
    if (stats->epc[i] != 0xBB) {
      return;
    }
  }

  if (output_files[2] == NULL) {
    output_files[2] = fopen(output_paths[2], "a");
    if (output_files[2] == NULL) {
      perror("Failed to open output file for Phase Angle vs Frequency");
      return;
    }
    setvbuf(output_files[2], NULL, _IOLBF, 0); // line-buffered
  }
  fprintf(output_files[2], "%u\t%d\n",
          get_channel_frequency(stats->channel_index),
          stats->rf_phase_angle);
}

void handle_bbbbbbbbbbbbbbbbbbbbbbbb_rssi16_freq(const epc_stats_t* stats) {
  for (int i = 0; i < 12; i++) {
    if (stats->epc[i] != 0xBB) {
      return;
    }
  }
  if (output_files[5] == NULL) {
    output_files[5] = fopen(output_paths[5], "a");
    if (output_files[5] == NULL) {
      perror("Failed to open output file for RSSI (16-bit) vs Frequency");
      return;
    }
    setvbuf(output_files[5], NULL, _IOLBF, 0); // line-buffered
  }
  fprintf(output_files[5], "%u\t%d\n",
          get_channel_frequency(stats->channel_index),
          stats->peak_rssi_16bit);
}

void handle_bbbbbbbbbbbbbbbbbbbbbbbb_doppler_freq(const epc_stats_t* stats) {
  for (int i = 0; i < 12; i++) {
    if (stats->epc[i] != 0xBB) {
      return;
    }
  }

  if (output_files[3] == NULL) {
    output_files[3] = fopen(output_paths[3], "a");
    if (output_files[3] == NULL) {
      perror("Failed to open output file for Doppler Frequency vs Frequency");
      return;
    }
    setvbuf(output_files[3], NULL, _IOLBF, 0); // line-buffered
  }
  fprintf(output_files[3], "%u\t%d\n",
          get_channel_frequency(stats->channel_index),
          stats->rf_doppler_frequency);
}

void handle_bbbbbbbbbbbbbbbbbbbbbbbb_phase_time(const epc_stats_t* stats) {
  for (int i = 0; i < 12; i++) {
    if (stats->epc[i] != 0xBB) {
      return;
    }
  }

  if (output_files[4] == NULL) {
    output_files[4] = fopen(output_paths[4], "a");
    if (output_files[4] == NULL) {
      perror("Failed to open output file for Phase Angle vs Time");
      return;
    }
    setvbuf(output_files[4], NULL, _IOLBF, 0); // line-buffered
  }
  fprintf(output_files[4], "%llu\t%d\n",
          (unsigned long long)stats->last_seen_timestamp_utc_microseconds,
          stats->rf_phase_angle);
}

void handle_bbbbbbbbbbbbbbbbbbbbbbbb_ccp_time(const epc_stats_t* stats) {
  
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

int fh_ready_flag = 0;

int main(void) {
  signal(SIGINT, signal_handler);
  setvbuf(stdout, NULL, _IOLBF, 0);  // line-buffered stdout
  setvbuf(stdin, NULL, _IOLBF, 0);  // line-buffered stdin
  prepare_output_dir();
  fh_init();
  char line[512];
  while (fgets(line, sizeof(line), stdin)) {
    line[strcspn(line, "\n")] = 0; // Remove newline
    //printf("Processing line: %s\n", line);
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
      handle_bbbbbbbbbbbbbbbbbbbbbbbb_rssi8_time(&stats);
      handle_bbbbbbbbbbbbbbbbbbbbbbbb_rssi16_time(&stats);
      handle_bbbbbbbbbbbbbbbbbbbbbbbb_phase_freq(&stats);
      handle_bbbbbbbbbbbbbbbbbbbbbbbb_doppler_freq(&stats);
      handle_bbbbbbbbbbbbbbbbbbbbbbbb_phase_time(&stats);
      handle_bbbbbbbbbbbbbbbbbbbbbbbb_rssi16_freq(&stats);
      handle_bbbbbbbbbbbbbbbbbbbbbbbb_ccp_time(&stats);
    }
  }
  fprintf(stderr, "Input stream closed. Exiting.\n");
  return 0;
}