#include <signal.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
//#include <inttypes.h>
#include "fh.h"
#include "mbuffer.h"

// #define MBUF_CAPTURE_WINDOW_SCALER 1
// //#define MBUF_FLUSH_INTERVAL 10
// #define MBUF_CAPTURE_WINDOW_DIV 8
// #define MBUF_CAPTURE_INTERVAL ((NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS) * NUM_ANTENNAS * NUM_CHANNELS * MBUF_CAPTURE_WINDOW_SCALER / MBUF_CAPTURE_WINDOW_DIV)


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
  //size_t ta_measurement_counts[NUM_ANTENNAS * (NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS)] = {0};
  size_t tag_channels_counts[NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS] = {0};
  size_t tag_antennas_counts[NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS] = {0};
  memset(tag_measurement_counts, 0, sizeof(tag_measurement_counts));
  //memset(ta_measurement_counts, 0, sizeof(ta_measurement_counts));
  memset(tag_channels_counts, 0, sizeof(tag_channels_counts));
  memset(tag_antennas_counts, 0, sizeof(tag_antennas_counts));

  uint8_t channels_usage[NUM_CHANNELS] = {0};
  memset(channels_usage, 0, sizeof(channels_usage));

  for (size_t i1 = 0; i1 < NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS; i1++) {
    size_t antenna_usage[NUM_ANTENNAS] = {0};
    memset(antenna_usage, 0, sizeof(antenna_usage));
    for (size_t j = 0; j < NUM_CHANNELS; j++) {
      int num_measurements = 0;
      for (size_t i0 = 0; i0 < NUM_ANTENNAS; i0++) {
        if (mbuf->data[i1 * NUM_ANTENNAS + i0].counter[j] > 0) {
          uint32_t n = mbuf->data[i1 * NUM_ANTENNAS + i0].counter[j];
          num_measurements += n;
          tag_measurement_counts[i1] += n;
          antenna_usage[i0] += n;
        }
      }
      if (num_measurements > 0) {
        tag_channels_counts[i1]++;
        channels_usage[j] += num_measurements;
      }
    }
    for (size_t j = 0; j < NUM_ANTENNAS; j++) {
      if (antenna_usage[j] > 0) {
        tag_antennas_counts[i1]++;
        num_measured_ta_actual++;
        printf("MBUFFER STATS: Tag-Antenna pair %zu,%zu has %zu measurements.\n", i1, j, antenna_usage[j]);
      } else {
        printf("MBUFFER STATS: Tag-Antenna pair %zu,%zu is missing measurements.\n", i1, j);
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

static void mbuffer_stats_extremes(mbuffer_t *mbuf) {
  static int16_t rssi_min=RSSI_MAX_VALUE, rssi_max=RSSI_MIN_VALUE;
  static uint16_t phase_angle_min=0xffff, phase_angle_max=0x0000;
  static int16_t doppler_frequency_min=INT16_MAX, doppler_frequency_max=INT16_MIN;
  static uint32_t counter_min=UINT32_MAX, counter_max=0;
  static double nrssi_min = 10.0, nrssi_max = -10.0;
  static double nphase_min = 10.0, nphase_max = -10.0;
  static double ndoppler_min = 10.0, ndoppler_max = -10.0;
  
  for (int i = 0; i < NUM_ANTENNAS * (NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS); i++) {
    mbuffer_unit_t unit = mbuf->data[i];
    mbuffer_normalized_unit_t nunit = mbuffer_get_normalized_unit(mbuf, (i % NUM_ANTENNAS) + 1, i / NUM_ANTENNAS);
    for (int j = 0; j < NUM_CHANNELS; j++) {
      for (int k = 0; k < NUM_MEASUREMENTS; k++) {
        int16_t rssi = unit.rssi[j][k];
        uint16_t phase_angle = unit.phase_angle[j][k];
        int16_t doppler_frequency = unit.doppler_frequency[j][k];
        uint32_t counter = unit.counter[j];
        if (rssi > rssi_max) rssi_max = rssi;
        if ((rssi < rssi_min) && (rssi != RSSI_MIN_VALUE)) rssi_min = rssi;
        if ((phase_angle > phase_angle_max) && (phase_angle != 0)) phase_angle_max = phase_angle;
        if ((phase_angle < phase_angle_min) && (phase_angle != 0)) phase_angle_min = phase_angle;
        if ((doppler_frequency > doppler_frequency_max) && (doppler_frequency != 0)) doppler_frequency_max = doppler_frequency;
        if ((doppler_frequency < doppler_frequency_min) && (doppler_frequency != 0)) doppler_frequency_min = doppler_frequency;
        if (counter > counter_max) counter_max = counter;
        if (counter < counter_min) counter_min = counter;

        double nrssi = nunit.rssi[j][k];
        double nphase = nunit.phase_angle[j][k];
        double ndoppler = nunit.doppler_frequency[j][k];
        if (nrssi > nrssi_max) nrssi_max = nrssi;
        if ((nrssi < nrssi_min) && (rssi != RSSI_MIN_VALUE)) nrssi_min = nrssi;
        if ((nphase > nphase_max) && (phase_angle != 0)) nphase_max = nphase;
        if ((nphase < nphase_min) && (phase_angle != 0)) nphase_min = nphase;
        if ((ndoppler > ndoppler_max) && (doppler_frequency != 0)) ndoppler_max = ndoppler;
        if ((ndoppler < ndoppler_min) && (doppler_frequency != 0)) ndoppler_min = ndoppler;

      }
    }
  }
  printf("Mbufer extremes stats:\n");
  printf("  rssi --> [%hd, %hd]\n", rssi_min, rssi_max);
  printf("  phase_angle --> [%hu, %hu]\n", phase_angle_min, phase_angle_max);
  printf("  doppler_frequency --> [%hd, %hd]\n", doppler_frequency_min, doppler_frequency_max);
  printf("  counter --> [%u, %u]\n", counter_min, counter_max);
  printf("  Normalized rssi --> [%f, %f]\n", nrssi_min, nrssi_max);
  printf("  Normalized phase angle --> [%f, %f]\n", nphase_min, nphase_max);
  printf("  Normalized doppler frequency --> [%f, %f]\n", ndoppler_min, ndoppler_max);
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
          if (normalized_unit.counter[i] > 0) {
            assert(0 < fprintf(file_rssi_tmp, "%f\n", normalized_unit.rssi[i][normalized_unit.counter[i] - 1]));
            assert(0 < fprintf(file_phase_tmp, "%f\n", normalized_unit.phase_angle[i][normalized_unit.counter[i] - 1]));
            assert(0 < fprintf(file_doppler_tmp, "%f\n", normalized_unit.doppler_frequency[i][normalized_unit.counter[i] - 1]));
          } else {
            assert(0 < fprintf(file_rssi_tmp, "%f\n", normalized_unit.rssi[i][0]));
            assert(0 < fprintf(file_phase_tmp, "%f\n", normalized_unit.phase_angle[i][0]));
            assert(0 < fprintf(file_doppler_tmp, "%f\n", normalized_unit.doppler_frequency[i][0]));
          }
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

#define NUM_FEATURES ((NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS) * NUM_ANTENNAS * NUM_MEASUREMENTS * 4)

static void handle_mbuffer_capture2(mbuffer_t *mbuf, uint16_t channel, const char* output_csv_file_path) {
  //printf("--> Channel: %hu\n", channel);
  uint16_t physical_channel = fh_get_physical_channel_index(channel);
  double features[NUM_FEATURES] = {0};
  for (int i = 0; i < (NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS) * NUM_ANTENNAS; i++) {
    mbuffer_normalized_unit_t nunit = mbuffer_get_normalized_unit(mbuf, (i % NUM_ANTENNAS) + 1, i / NUM_ANTENNAS);
    for (int j = 0; j < NUM_MEASUREMENTS; j++) {
      double* measurement_features = &features[i * NUM_MEASUREMENTS * 4 + j];
      measurement_features[0] = nunit.rssi[physical_channel][j];
      measurement_features[1] = nunit.phase_angle[physical_channel][j];
      measurement_features[2] = nunit.phase_angle[physical_channel][j];
      measurement_features[3] = 2.0 * (((double)physical_channel) / ((double)(NUM_CHANNELS - 1))) - 1.0;
    }
  }

  FILE *fp = fopen(output_csv_file_path, "a");
  if (fp != NULL) {
    for (int i = 0; i < NUM_FEATURES; i++) {
      fprintf(fp, "%.6f", features[i]);
      if (i < NUM_FEATURES - 1) {
        fprintf(fp, ",");
      }
    }
    fprintf(fp, "\n");
    fclose(fp);
  }
}

int main(int argc, char** argv) {
  signal(SIGINT, signal_handler);
  setvbuf(stdout, NULL, _IOLBF, 0);  // line-buffered stdout
  setvbuf(stdin, NULL, _IOLBF, 0);  // line-buffered stdin

  if (argc < 2) {
    fprintf(stderr, "Error: Missing output CSV file path.\n");
    return 1;
  } else if (argc != 2) {
    fprintf(stderr, "Error: Extra arguments detected. The only argument needed is the output CSV file path.\n");
    return 1;
  }

  const char* output_csv_file_path = argv[1];
  if (access(output_csv_file_path, F_OK) == 0) {
    fprintf(stderr, "Error: File '%s' already exists. Please specify a different output CSV file path.\n", output_csv_file_path);
    return 1;
  }

  fh_init();
  mbuffer_t mbuf;
  mbuffer_flush(&mbuf);

  char line[512];
  uint32_t mcounter = 0;
  uint64_t window_counter = 0;
  int channel = -1;
  int antenna = -1;
  int num_antennas_traversed = 0;
  time_t t = time(NULL);
  uint64_t num_windows_this_second = 0;
  uint64_t num_measurements_this_second = 0;
  int window_acquisition_rate = 0;
  int measurement_acquisition_rate = 0;
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

      if ((!((channel == -1) && (antenna == -1))) && ((channel != stats.channel_index) || (antenna != stats.antenna_id))) {
        num_antennas_traversed++;
      }

      num_measurements_this_second++;

      // if (mcounter % MBUF_CAPTURE_INTERVAL == 0) {
      //if ((!((channel == -1) && (antenna == -1))) && ((channel != stats.channel_index) || (antenna != stats.antenna_id))) {
      if (num_antennas_traversed == NUM_ANTENNAS) {
      //if ((channel != -1) && (channel != stats.channel_index)) {
        num_antennas_traversed = 0;
        // Capture mbuffer state

        //printf("Capturing mbuffer state after %u measurements.\n", MBUF_CAPTURE_INTERVAL);
        printf("Capturing mbuffer state after %u measurements.\n", mcounter);
        mbuffer_stats(&mbuf);
        //mbuffer_stats_extremes(&mbuf);
        window_counter++;
        num_windows_this_second++;
        printf("Window counter: %llu\n", window_counter);
        time_t now = time(NULL);
        if (now > t) {
          window_acquisition_rate = num_windows_this_second;
          measurement_acquisition_rate = num_measurements_this_second;
          t = time(NULL);
          num_windows_this_second = 0;
          num_measurements_this_second = 0;
        }
        printf("Window acquisition rate: %llu Hz\n", window_acquisition_rate);
        printf("Measurement acquisition rate: %llu Hz\n", measurement_acquisition_rate);
        //handle_mbuffer_capture(&mbuf);
        handle_mbuffer_capture2(&mbuf, channel, output_csv_file_path);
        // if (mcounter % MBUF_FLUSH_INTERVAL == 0) {
        //   printf("Flushing mbuffer after %u measurements.\n", MBUF_FLUSH_INTERVAL * MBUF_CAPTURE_INTERVAL);
           mbuffer_flush(&mbuf);
        // }
        mcounter = 0;
      }
      channel = stats.channel_index;
      antenna = stats.antenna_id;

      if (tag_index >= NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS) {
        printf("Error: unexpected tag index %u (max %u). Tag EPC is: %02X%02X%02X%02X%02X%02X%02X%02X%02X%02X%02X%02X\n",
          tag_index, NUM_REFERENCE_TAGS + NUM_TRACKED_ASSETS - 1,
          stats.epc[0], stats.epc[1], stats.epc[2], stats.epc[3], stats.epc[4], stats.epc[5],
          stats.epc[6], stats.epc[7], stats.epc[8], stats.epc[9], stats.epc[10], stats.epc[11]);
      }
      mbuffer_update_unit(&mbuf, (uint8_t)stats.antenna_id, tag_index, stats.channel_index,
                          stats.peak_rssi_16bit, stats.rf_phase_angle, stats.rf_doppler_frequency);
      mcounter++;
    }
    
  }
  assert(0 < fprintf(stderr, "Input stream closed. Exiting.\n"));
  return 0;
}