#include <stdio.h>
#include "visualizer.h"
#include "../telemetry.h"

int main() {
  Visualizer_DisplayVisualizationWindow();
  struct telemetry t = telemetry_init_server(8080);
  telemetry_connect(&t);
  telemetry_print_sockopts(&t);

  struct telemetry_packet packet;
  while(1) {
    telemetry_receive(&t, &packet);
    printf("EPC: ");
    for (int i = 0; i < 12; i++) {
      printf("%02X", packet.epc[i]);
    }
    printf(", RSSI0: %d, RSSI1: %d\n", packet.rssi0, packet.rssi1);
  }
  return 0;
}