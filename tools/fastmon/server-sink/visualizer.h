#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <stdint.h>
#include <SDL2/SDL.h>

#define VISUALIZER_DATA_CONTEXT_MAX_TAGS 100

typedef struct Visualizer Visualizer_t;
typedef struct Visualizer_ColorRGBA {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
} Visualizer_ColorRGBA_t;

Visualizer_t* Visualizer_Create();
void Visualizer_Destroy(Visualizer_t* visualizer);

void Visualizer_DrawCircle(Visualizer_t* visualizer, int32_t centreX, int32_t centreY, int32_t radius, Visualizer_ColorRGBA_t color);
void Visualizer_DrawHollowCircle(Visualizer_t* visualizer, int32_t centreX, int32_t centreY, int32_t radius, Visualizer_ColorRGBA_t color);
int Visualizer_DisplayVisualizationWindow(Visualizer_t* visualizer);
void Visualizer_Coordinates_Unit2Absolute(Visualizer_t* visualizer, float xUnit_in, float yUnit_in, int32_t* xAbsolute_out, int32_t* yAbsolute_out);
unsigned int Visualizer_GetWindowWidth(Visualizer_t* visualizer);
unsigned int Visualizer_GetWindowHeight(Visualizer_t* visualizer);
void Visualizer_Render(Visualizer_t* visualizer);

void Visualizer_Delay(Visualizer_t* visualizer, uint32_t ms);

#endif // VISUALIZER_H