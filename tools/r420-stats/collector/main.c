#include <signal.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include "fh.h"
#include "mbuffer.h"

#define MBUF_CAPTURE_WINDOW_SCALER 1
//#define MBUF_FLUSH_INTERVAL 10
#define MBUF_CAPTURE_WINDOW_DIV 8
#define MBUF_CAPTURE_INTERVAL ((NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS) * NUM_ANTENNAS * NUM_CHANNELS * MBUF_CAPTURE_WINDOW_SCALER / MBUF_CAPTURE_WINDOW_DIV)


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

static void signal_handler(int sig){
  if (sig == SIGINT) {
    printf("Caught SIGINT, exiting.\n");
  }
}

static void mbuffer_stats(mbuffer_t *mbuf) {
  // Compute and print basic statistics of the mbuffer
  size_t num_measured_ta_ideal = NUM_ANTENNAS * (NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS);
  size_t num_measured_ta_actual = 0;

  size_t tag_measurement_counts[NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS] = {0};
  size_t tag_channels_counts[NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS] = {0};
  size_t tag_antennas_counts[NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS] = {0};
  memset(tag_measurement_counts, 0, sizeof(tag_measurement_counts));
  memset(tag_channels_counts, 0, sizeof(tag_measurement_counts));
  memset(tag_antennas_counts, 0, sizeof(tag_antennas_counts));

  uint8_t channels_usage[NUM_CHANNELS] = {0};
  memset(channels_usage, 0, sizeof(channels_usage));

  for (size_t i = 0; i < NUM_ANTENNAS * (NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS); i++) {
    for (size_t j = 0; j < NUM_CHANNELS; j++) {
      if (mbuf->data[i].counter[j] > 0) {
        if (tag_measurement_counts[i / NUM_ANTENNAS] == 0) {
          num_measured_ta_actual++;
        }
        tag_channels_counts[i / NUM_ANTENNAS]++;
        tag_measurement_counts[i / NUM_ANTENNAS] += mbuf->data[i].counter[j];
        channels_usage[j]++;
      }
      if (j == NUM_CHANNELS - 1) {
        if (tag_measurement_counts[i / NUM_ANTENNAS] == 0) {
          // No measurements for this tag-antenna pair
          printf("MBUFFER STATS: Tag-Antenna pair %zu is missing measurements.\n", i);
        } else {
          printf("MBUFFER STATS: Tag-Antenna pair %zu has %zu measurements.\n", i, tag_measurement_counts[i / NUM_ANTENNAS]);
          tag_antennas_counts[i / NUM_ANTENNAS]++;
        }
      }
    }
  }
  printf("MBUFFER STATS: Measured Tag-Antenna pairs: %zu / %zu\n",
         num_measured_ta_actual, num_measured_ta_ideal);
  for (size_t tag_index = 0; tag_index < NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS; tag_index++) {
    printf("MBUFFER STATS: Tag %zu measurement counts across antennas & channels: ", tag_index);
    printf("%zu ", tag_measurement_counts[tag_index]);
    printf(" (%zu channels, %zu antennas)", tag_channels_counts[tag_index], tag_antennas_counts[tag_index]);
    printf("\n");
  }
  for (size_t channel_index = 1; channel_index <= NUM_CHANNELS; channel_index++) {
    // printf("MBUFFER STATS: Channel %zu measurement counts across antennas & tags: ", channel_index);
    // printf("%zu ", channels_usage[channel_index]);
    // //printf(" (%zu times)", tag_channels_counts[tag_index]);
    // printf("\n");
    uint16_t physical_channel_index = fh_get_physical_channel_index(channel_index);
    if (channels_usage[physical_channel_index] > 0) {
      printf("🟩");
    } else {
      printf("🟥");
    }
  }
  printf("\n");
  for (size_t channel_index = 1; channel_index <= NUM_CHANNELS; channel_index++) {
    // printf("MBUFFER STATS: Channel %zu measurement counts across antennas & tags: ", channel_index);
    // printf("%zu ", channels_usage[channel_index]);
    // //printf(" (%zu times)", tag_channels_counts[tag_index]);
    // printf("\n");
    if (channels_usage[channel_index] > 0) {
      printf("🟦");
    } else {
      printf("🟥");//⬜
    }
  }
  printf("\n");
}

static void handle_mbuffer_capture(mbuffer_t *mbuf) {
  // for each (tag, antenna, metric) tuple, output recorded frequency-domain data to file `outputs/mbuf_<tag_index>_<antenna_index>_<metric>.csv` (replace file contents)
  for (uint32_t tag_index = 0; tag_index < NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS; tag_index++) {
    for (uint8_t antenna_id = 1; antenna_id <= NUM_ANTENNAS; antenna_id++) {
      mbuffer_normalized_unit_t normalized_unit = mbuffer_get_normalized_unit(mbuf, antenna_id, tag_index);
      // Output to files
      char filename_rssi[256];
      char filename_phase[256];
      char filename_doppler[256];
      char filename_rssi_tmp[256];
      char filename_phase_tmp[256];
      char filename_doppler_tmp[256];
      assert(0 < snprintf(filename_rssi, sizeof(filename_rssi), "outputs/mbuf_%u_%u_rssi.csv", tag_index, antenna_id));
      assert(0 < snprintf(filename_phase, sizeof(filename_phase), "outputs/mbuf_%u_%u_phase.csv", tag_index, antenna_id));
      assert(0 < snprintf(filename_doppler, sizeof(filename_doppler), "outputs/mbuf_%u_%u_doppler.csv", tag_index, antenna_id));
      assert(0 < snprintf(filename_rssi_tmp, sizeof(filename_rssi_tmp), "outputs/mbuf_%u_%u_rssi.csv.tmp", tag_index, antenna_id));
      assert(0 < snprintf(filename_phase_tmp, sizeof(filename_phase_tmp), "outputs/mbuf_%u_%u_phase.csv.tmp", tag_index, antenna_id));
      assert(0 < snprintf(filename_doppler_tmp, sizeof(filename_doppler_tmp), "outputs/mbuf_%u_%u_doppler.csv.tmp", tag_index, antenna_id));
      // Write to temp files first
      FILE *file_rssi_tmp = fopen(filename_rssi_tmp, "w");
      FILE *file_phase_tmp = fopen(filename_phase_tmp, "w");
      FILE *file_doppler_tmp = fopen(filename_doppler_tmp, "w");
      if (file_rssi_tmp && file_phase_tmp && file_doppler_tmp) {
        for (size_t i = 0; i < NUM_CHANNELS; i++) {
          assert(0 < fprintf(file_rssi_tmp, "%f\n", normalized_unit.rssi[i]));
          assert(0 < fprintf(file_phase_tmp, "%f\n", normalized_unit.phase_angle[i]));
          assert(0 < fprintf(file_doppler_tmp, "%f\n", normalized_unit.doppler_frequency[i]));
        }
      } else {
        assert(0 < fprintf(stderr, "Error opening output files for tag %u, antenna %u (errno: %d)\n",
          tag_index, antenna_id, errno));
      }
      if (file_rssi_tmp) fclose(file_rssi_tmp);
      if (file_phase_tmp) fclose(file_phase_tmp);
      if (file_doppler_tmp) fclose(file_doppler_tmp);
      // Rename temp files to final filenames
      if (rename(filename_rssi_tmp, filename_rssi) != 0) {
        assert(0 < fprintf(stderr, "Error renaming rssi file for tag %u, antenna %u (errno: %d)\n", tag_index, antenna_id, errno));
      }
      if (rename(filename_phase_tmp, filename_phase) != 0) {
        assert(0 < fprintf(stderr, "Error renaming phase file for tag %u, antenna %u (errno: %d)\n", tag_index, antenna_id, errno));
      }
      if (rename(filename_doppler_tmp, filename_doppler) != 0) {
        assert(0 < fprintf(stderr, "Error renaming doppler file for tag %u, antenna %u (errno: %d)\n", tag_index, antenna_id, errno));
      }
    }
  }
}

int main(void) {
  signal(SIGINT, signal_handler);
  setvbuf(stdout, NULL, _IOLBF, 0);  // line-buffered stdout
  setvbuf(stdin, NULL, _IOLBF, 0);  // line-buffered stdin
  fh_init();
  mbuffer_t mbuf;
  mbuffer_flush(&mbuf);

  char line[512];
  uint32_t mcounter = 0;
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
      // printf("Measurement entry | EPC: %02X%02X%02X%02X%02X%02X%02X%02X%02X%02X%02X%02X, Antenna ID: %u, Peak RSSI: %d, RF Phase Angle: %d, RF Doppler Frequency: %d, 16-bit peak rssi: %d, First Seen Timestamp (UTC): %llu us, Last Seen Timestamp (UTC): %llu us, Tag Seen Count: %u, Channel Index: %u\n",
      //   stats.epc[0], stats.epc[1], stats.epc[2], stats.epc[3], stats.epc[4], stats.epc[5],
      //   stats.epc[6], stats.epc[7], stats.epc[8], stats.epc[9], stats.epc[10], stats.epc[11],
      //   stats.antenna_id, stats.peak_rssi, stats.rf_phase_angle, stats.rf_doppler_frequency,
      //   stats.peak_rssi_16bit, (unsigned long long)stats.first_seen_timestamp_utc_microseconds,
      //   (unsigned long long)stats.last_seen_timestamp_utc_microseconds, stats.tag_seen_count, stats.channel_index);

      if (((uint32_t)stats.epc[0] << 24 |
          (uint32_t)stats.epc[1] << 16 |
          (uint32_t)stats.epc[2] << 8 |
          (uint32_t)stats.epc[3]) != 0x14B00473) {
        printf("Unknown tag (non-labmate prefix): %02X,%02X,%02X,%02X|%02X%02X%02X%02X%02X%02X%02X%02X\n",
          stats.epc[0], stats.epc[1], stats.epc[2], stats.epc[3], stats.epc[4], stats.epc[5],
          stats.epc[6], stats.epc[7], stats.epc[8], stats.epc[9], stats.epc[10], stats.epc[11]);
        continue; // Unknown tag
      }

      uint32_t tag_index = ((uint32_t)stats.epc[8]) << 24 |
                           ((uint32_t)stats.epc[9]) << 16 |
                           ((uint32_t)stats.epc[10]) << 8 |
                           ((uint32_t)stats.epc[11]);

      if (((uint32_t)stats.epc[4] << 24 |
          (uint32_t)stats.epc[5] << 16 |
          (uint32_t)stats.epc[6] << 8 |
          (uint32_t)stats.epc[7]) == 0x00000000) {
          tag_index += NUM_REFERENCE_TAGS; // Tracked asset
      } else if (((uint32_t)stats.epc[4] << 24 |
          (uint32_t)stats.epc[5] << 16 |
          (uint32_t)stats.epc[6] << 8 |
          (uint32_t)stats.epc[7]) != 0xDEADBEEF) {
        printf("Unknown tag : %02X%02X%02X%02X%02X%02X%02X%02X%02X%02X%02X%02X\n",
          stats.epc[0], stats.epc[1], stats.epc[2], stats.epc[3], stats.epc[4], stats.epc[5],
          stats.epc[6], stats.epc[7], stats.epc[8], stats.epc[9], stats.epc[10], stats.epc[11]);
        continue; // Unknown tag
      }

      mbuffer_update_unit(&mbuf, (uint8_t)stats.antenna_id, tag_index, stats.channel_index,
                          stats.peak_rssi_16bit, stats.rf_phase_angle, stats.rf_doppler_frequency);
      mcounter++;
      if (mcounter % MBUF_CAPTURE_INTERVAL == 0) {
        // Capture mbuffer state

        printf("Capturing mbuffer state after %u measurements.\n", MBUF_CAPTURE_INTERVAL);
        mbuffer_stats(&mbuf);
        handle_mbuffer_capture(&mbuf);
        // if (mcounter % MBUF_FLUSH_INTERVAL == 0) {
        //   printf("Flushing mbuffer after %u measurements.\n", MBUF_FLUSH_INTERVAL * MBUF_CAPTURE_INTERVAL);
           mbuffer_flush(&mbuf);
        // }
        mcounter = 0;
      }
    }
    
  }
  assert(0 < fprintf(stderr, "Input stream closed. Exiting.\n"));
  return 0;
}