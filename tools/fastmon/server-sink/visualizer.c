#include "visualizer.h"
#include <SDL2/SDL_ttf.h>
#include <pthread.h>
#include <math.h>
#include "config.h"

static SDL_Window   *m_window          = NULL;
static SDL_Renderer *m_window_renderer = NULL;
static SDL_Event     m_window_event;
static unsigned int window_width = VISUALIZER_WINDOW_WIDTH;
static unsigned int window_height = VISUALIZER_WINDOW_HEIGHT;

/**
 * @todo Enhance interpolation
 * @todo Color-code time left to serve customer
*/

void Visualizer_DrawCircle(SDL_Renderer* renderer, int32_t centreX, int32_t centreY, int32_t radius) {
    for (int32_t y = -radius; y <= radius; y++) {
        for (int32_t x = -radius; x <= radius; x++) {
            if (x*x + y*y <= radius*radius) {
                SDL_RenderDrawPoint(renderer, centreX + x, centreY + y);
            }
        }
    }
}

/**
 * @source https://discourse.libsdl.org/t/query-how-do-you-draw-a-circle-in-sdl2-sdl2/33379
*/
void Visualizer_DrawHollowCircle(SDL_Renderer* renderer, int32_t centreX, int32_t centreY, int32_t radius) {
    const int32_t diameter = (radius * 2);

    int32_t x = (radius - 1);
    int32_t y = 0;
    int32_t tx = 1;
    int32_t ty = 1;
    int32_t error = (tx - diameter);

    while (x >= y) {
        // Each of the following renders an octant of the circle
        SDL_RenderDrawPoint(renderer, centreX + x, centreY - y);
        SDL_RenderDrawPoint(renderer, centreX + x, centreY + y);
        SDL_RenderDrawPoint(renderer, centreX - x, centreY - y);
        SDL_RenderDrawPoint(renderer, centreX - x, centreY + y);
        SDL_RenderDrawPoint(renderer, centreX + y, centreY - x);
        SDL_RenderDrawPoint(renderer, centreX + y, centreY + x);
        SDL_RenderDrawPoint(renderer, centreX - y, centreY - x);
        SDL_RenderDrawPoint(renderer, centreX - y, centreY + x);

        if (error <= 0) {
      	    ++y;
      	    error += ty;
      	    ty += 2;
        }

        if (error > 0) {
      	    --x;
      	    tx += 2;
      	    error += (tx - diameter);
        }
    }
}

static void* eventListenerThreadHandler(void* arg_p) {
    while(1)
    {
        SDL_WaitEvent(&m_window_event);
        //while(SDL_PollEvent(&m_window_event) > 0)
        switch(m_window_event.type) {
            case SDL_QUIT:
                SDL_DestroyRenderer(m_window_renderer);
                SDL_DestroyWindow(m_window);
                exit(0);
        }
        //update(1.0/60.0, &x, &y);
        //draw(m_window_renderer, x, y);
    }
}

static int init_ttf() {
  if (TTF_Init() < 0) {
      printf("TTF_Init: %s\n", TTF_GetError());
      return 1;
  }
  return 0;
}

int Visualizer_DisplayVisualizationWindow() {

  m_window = SDL_CreateWindow("Visualizer",
                              0/*SDL_WINDOWPOS_CENTERED*/,
                              0/*SDL_WINDOWPOS_CENTERED*/,
                              VISUALIZER_WINDOW_WIDTH, VISUALIZER_WINDOW_HEIGHT,
                              0);

  if(m_window == NULL)
  {
      printf("Failed to create window\n");
      printf("SDL2 Error: %s\n", SDL_GetError());
      return 1;
  }

  //unsigned int window_width = VISUALIZER_WINDOW_WIDTH;
  //unsigned int window_height = VISUALIZER_WINDOW_HEIGHT;
  if (VISUALIZER_WINDOW_WIDTH == 0 || VISUALIZER_WINDOW_HEIGHT == 0) {
      SDL_DisplayMode dm;
      if (SDL_GetDesktopDisplayMode(0, &dm))
      {
          printf("Error getting desktop display mode\n");
          return -1;
      }
      window_width = VISUALIZER_WINDOW_WIDTH == 0 ? dm.w : VISUALIZER_WINDOW_WIDTH;
      window_height = VISUALIZER_WINDOW_HEIGHT == 0 ? dm.h : VISUALIZER_WINDOW_HEIGHT;;
  }
  
  SDL_SetWindowSize(m_window, window_width, window_height);

  
  m_window_renderer = SDL_CreateRenderer(m_window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

  if(m_window_renderer == NULL)
  {
      printf("Failed to create renderer\n");
      printf("SDL2 Error: %s\n", SDL_GetError());
      return 1;
  }

  init_ttf();
  
  pthread_t eventListenerTID;
  pthread_create(&eventListenerTID, NULL, eventListenerThreadHandler, NULL);

  return 0;
}