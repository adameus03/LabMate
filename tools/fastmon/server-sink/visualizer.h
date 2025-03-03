#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <stdint.h>
#include <SDL2/SDL.h>

#define VISUALIZER_DATA_CONTEXT_MAX_TAGS 100

typedef struct visualizer_data_context {
  uint32_t ids[VISUALIZER_DATA_CONTEXT_MAX_TAGS];
  uint8_t rssi0[VISUALIZER_DATA_CONTEXT_MAX_TAGS];
  uint8_t rssi1[VISUALIZER_DATA_CONTEXT_MAX_TAGS];
} visualizer_data_context_t;

void Visualizer_DrawCircle(SDL_Renderer* renderer, int32_t centreX, int32_t centreY, int32_t radius);
void Visualizer_DrawHollowCircle(SDL_Renderer* renderer, int32_t centreX, int32_t centreY, int32_t radius);
int Visualizer_DisplayVisualizationWindow();

#endif // VISUALIZER_H