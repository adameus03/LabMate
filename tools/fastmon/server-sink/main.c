#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include "visualizer.h"
#include "../telemetry.h"

#define EPC_LEN 12
#define EPC_ENTRIES_MAX 100
#define PORT 8080
#define DOT_RADIUS 5

typedef struct epc_color_entry {
  uint8_t epc[EPC_LEN];
  Visualizer_ColorRGBA_t color;
} epc_color_entry_t;

typedef struct epc_color_dictionary {
  epc_color_entry_t entries[EPC_ENTRIES_MAX];
  int n_entries;
} epc_color_dictionary_t;

static Visualizer_ColorRGBA_t main_colorRGBA_random() {
  Visualizer_ColorRGBA_t color;
  color.r = rand() % 256;
  color.g = rand() % 256;
  color.b = rand() % 256;
  color.a = 255;
  return color;
}

static int main_epc_color_dictionary_GetPosition(const epc_color_dictionary_t* epc_color_dictionary, const uint8_t* epc) {
  assert(epc_color_dictionary != NULL);
  assert(epc != NULL);
  for (int i = 0; i < epc_color_dictionary->n_entries; i++) {
    if (memcmp(epc_color_dictionary->entries[i].epc, epc, EPC_LEN) == 0) {
      return i;
    }
  }
  return -1;
}

static Visualizer_ColorRGBA_t main_epc_get_color(const char* epc, epc_color_dictionary_t* epc_color_dictionary) {
  assert(epc != NULL);
  assert(epc_color_dictionary != NULL);
  int pos = main_epc_color_dictionary_GetPosition(epc_color_dictionary, epc);
  if (pos == -1) {
    if (! (epc_color_dictionary->n_entries < EPC_ENTRIES_MAX)) {
      printf("main_epc_get_color: EPC dictionary full\n");
      exit(1);
    }
    pos = epc_color_dictionary->n_entries;
    epc_color_dictionary->n_entries++;
    assert(memcpy(epc_color_dictionary->entries[pos].epc, epc, EPC_LEN) == epc_color_dictionary->entries[pos].epc);
    epc_color_dictionary->entries[pos].color = main_colorRGBA_random();
  }
  printf("Color buffer size: %d / %d\n", epc_color_dictionary->n_entries, EPC_ENTRIES_MAX);
  return epc_color_dictionary->entries[pos].color;
}

static void main_rssi2unit(uint8_t rssi_in, float* rssi_out) {
  assert(rssi_out != NULL);
  assert(rssi_in <= 200 && (rssi_in >= 180 || rssi_in == 0));
  if (rssi_in == 0) {
    *rssi_out = 0.0f;
  } else {
    *rssi_out = ((float)(rssi_in - 180)) / 20.0f;
  }
}

int main() {
  srand(time(NULL));
  Visualizer_t* visualizer = Visualizer_Create();
  Visualizer_DisplayVisualizationWindow(visualizer);
  struct telemetry t = telemetry_init_server(PORT);
  telemetry_connect(&t);
  telemetry_print_sockopts(&t);

  epc_color_dictionary_t epc_color_dictionary;
  epc_color_dictionary.n_entries = 0;

  struct telemetry_packet packet;
  while(1) {
    telemetry_receive(&t, &packet);
    printf("EPC: ");
    for (int i = 0; i < 12; i++) {
      printf("%02X", packet.epc[i]);
    }
    float rssi0f = -1.0f;
    float rssi1f = -1.0f;
    main_rssi2unit(packet.rssi0, &rssi0f);
    main_rssi2unit(packet.rssi1, &rssi1f);
    assert(rssi0f >= 0.0f && rssi0f <= 1.0f);
    assert(rssi1f >= 0.0f && rssi1f <= 1.0f);
    printf(", RSSI0: %d, RSSI1: %d, RSSI0f: %f, RSSI1f: %f\n", packet.rssi0, packet.rssi1, rssi0f, rssi1f);
    int32_t x_absolute = -1;
    int32_t y_absolute = -1;
    Visualizer_Coordinates_Unit2Absolute(visualizer, rssi0f, rssi1f, &x_absolute, &y_absolute);
    assert(x_absolute >= 0 && x_absolute <= Visualizer_GetWindowWidth(visualizer));
    assert(y_absolute >= 0 && y_absolute <= Visualizer_GetWindowHeight(visualizer));
    
    if (x_absolute == 0 || y_absolute == 0) {
      printf("main: x_absolute or y_absolute is 0. Skipping packet\n");
      continue;
    }

    Visualizer_ColorRGBA_t color = main_epc_get_color(packet.epc, &epc_color_dictionary);

    // int32_t x_absolute = rand() % Visualizer_GetWindowWidth(visualizer);
    // int32_t y_absolute = rand() % Visualizer_GetWindowHeight(visualizer);
    // Visualizer_ColorRGBA_t color = main_colorRGBA_random();

    Visualizer_DrawCircle(visualizer, x_absolute, y_absolute, DOT_RADIUS, color);
    Visualizer_Render(visualizer);
    Visualizer_Delay(visualizer, 25);
  }
  Visualizer_Destroy(visualizer);
  telemetry_close(&t);
  return 0;
}