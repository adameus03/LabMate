#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

/*
  Relevant log lines begin with "R420 Log: "
  Example log line:
  R420 Log: Tag EPC: 30395DFA82F8960000291BE9, Antenna ID: 1, Peak RSSI: 196, RF Phase Angle: 804, RF Doppler Frequency: -273, 16-bit peak rssi: -6050, First Seen Timestamp (UTC): 1762871268217261 us, Last Seen Timestamp (UTC): 1762871268217261 us, Tag Seen Count: 1, Channel Index: 10
*/

#define TOTAL_CHANNELS 50
const uint32_t freq_hop_table[TOTAL_CHANNELS] = {
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

/**
 * @param channel_index Channel index (1-based)
 * @return Frequency in kHz
 */
uint32_t get_channel_frequency(uint8_t channel_index) {
  assert(channel_index >= 1);
  assert(channel_index < TOTAL_CHANNELS);
  return freq_hop_table[channel_index - 1];
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

int main(void) {
  setvbuf(stdout, NULL, _IOLBF, 0);  // line-buffered stdout
  setvbuf(stdin, NULL, _IOLBF, 0);  // line-buffered stdin
  char line[512];
  while (fgets(line, sizeof(line), stdin)) {
    line[strcspn(line, "\n")] = 0; // Remove newline
    printf("Processing line: %s\n", line);
    //TODO
    fflush(stdout);
  }
  fprintf(stderr, "Input stream closed. Exiting.\n");
  return 0;
}