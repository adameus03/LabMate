#include "visualizer.h"
#include <SDL2/SDL_ttf.h>
#include <pthread.h>
#include <math.h>
#include <assert.h>
#include "config.h"

struct Visualizer {
    SDL_Window* m_window;
    SDL_Renderer* m_window_renderer;
    SDL_Event m_window_event;
    unsigned int window_width;
    unsigned int window_height;
};

Visualizer_t* Visualizer_Create() {
    Visualizer_t* visualizer = (Visualizer_t*)malloc(sizeof(Visualizer_t));
    visualizer->m_window = NULL;
    visualizer->m_window_renderer = NULL;
    visualizer->window_width = VISUALIZER_WINDOW_WIDTH;
    visualizer->window_height = VISUALIZER_WINDOW_HEIGHT;
    return visualizer;
}

void Visualizer_Destroy(Visualizer_t* visualizer) {
    free(visualizer);
}

static void Visualizer_AssertSanity(Visualizer_t* visualizer) {
    if (visualizer == NULL) {
        printf("Visualizer_AssertSanity: visualizer is NULL\n");
        exit(1);
    }
    if (visualizer->m_window == NULL) {
        printf("Visualizer_AssertSanity: m_window is NULL\n");
        exit(1);
    }
    if (visualizer->m_window_renderer == NULL) {
        printf("Visualizer_AssertSanity: m_window_renderer is NULL\n");
        exit(1);
    }
}

static void Visualizer_SDLErrorCheck(int code) {
    if (code < 0) {
        const char* error = SDL_GetError();
        if (strlen(error) > 0) {
            printf("SDL Error: %s\n", error);
            exit(1);
        } else {
            printf("SDL Error: %d\n", code);
            exit(1);
        }
    }
}

static void Visualizer_SDLErrorCheck_Ptr(void* ptr) {
    if (ptr == NULL) {
        const char* error = SDL_GetError();
        if (strlen(error) > 0) {
            printf("SDL Error: %s\n", error);
            exit(1);
        } else {
            printf("SDL Error: NULL\n");
            exit(1);
        }
    }
}

/**
 * @todo Enhance interpolation
 * @todo Color-code time left to serve customer
*/

void Visualizer_DrawCircle(Visualizer_t* visualizer, int32_t centreX, int32_t centreY, int32_t radius, Visualizer_ColorRGBA_t color) {
    Visualizer_AssertSanity(visualizer);
    SDL_Renderer* renderer = visualizer->m_window_renderer;
    Visualizer_SDLErrorCheck(SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a));
    for (int32_t y = -radius; y <= radius; y++) {
        for (int32_t x = -radius; x <= radius; x++) {
            if (x*x + y*y <= radius*radius) {
                Visualizer_SDLErrorCheck(SDL_RenderDrawPoint(renderer, centreX + x, centreY + y));
            }
        }
    }
}

/**
 * @source https://discourse.libsdl.org/t/query-how-do-you-draw-a-circle-in-sdl2-sdl2/33379
*/
void Visualizer_DrawHollowCircle(Visualizer_t* visualizer, int32_t centreX, int32_t centreY, int32_t radius, Visualizer_ColorRGBA_t color) {
    Visualizer_AssertSanity(visualizer);
    SDL_Renderer* renderer = visualizer->m_window_renderer;
    Visualizer_SDLErrorCheck(SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a));
    const int32_t diameter = (radius * 2);

    int32_t x = (radius - 1);
    int32_t y = 0;
    int32_t tx = 1;
    int32_t ty = 1;
    int32_t error = (tx - diameter);

    while (x >= y) {
        // Each of the following renders an octant of the circle
        Visualizer_SDLErrorCheck(SDL_RenderDrawPoint(renderer, centreX + x, centreY - y));
        Visualizer_SDLErrorCheck(SDL_RenderDrawPoint(renderer, centreX + x, centreY + y));
        Visualizer_SDLErrorCheck(SDL_RenderDrawPoint(renderer, centreX - x, centreY - y));
        Visualizer_SDLErrorCheck(SDL_RenderDrawPoint(renderer, centreX - x, centreY + y));
        Visualizer_SDLErrorCheck(SDL_RenderDrawPoint(renderer, centreX + y, centreY - x));
        Visualizer_SDLErrorCheck(SDL_RenderDrawPoint(renderer, centreX + y, centreY + x));
        Visualizer_SDLErrorCheck(SDL_RenderDrawPoint(renderer, centreX - y, centreY - x));
        Visualizer_SDLErrorCheck(SDL_RenderDrawPoint(renderer, centreX - y, centreY + x));

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
    Visualizer_t* visualizer = (Visualizer_t*)arg_p;
    Visualizer_AssertSanity(visualizer);
    while(1)
    {
        Visualizer_SDLErrorCheck(SDL_WaitEvent(&visualizer->m_window_event));
        //while(SDL_PollEvent(&m_window_event) > 0)
        switch(visualizer->m_window_event.type) {
            case SDL_QUIT:
                SDL_DestroyRenderer(visualizer->m_window_renderer);
                SDL_DestroyWindow(visualizer->m_window);
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

int Visualizer_DisplayVisualizationWindow(Visualizer_t* visualizer) {
  assert(visualizer != NULL);
  visualizer->m_window = SDL_CreateWindow("Visualizer",
                              0/*SDL_WINDOWPOS_CENTERED*/,
                              0/*SDL_WINDOWPOS_CENTERED*/,
                              VISUALIZER_WINDOW_WIDTH, VISUALIZER_WINDOW_HEIGHT,
                              0);

  if(visualizer->m_window == NULL)
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
      visualizer->window_width = VISUALIZER_WINDOW_WIDTH == 0 ? dm.w : VISUALIZER_WINDOW_WIDTH;
      visualizer->window_height = VISUALIZER_WINDOW_HEIGHT == 0 ? dm.h : VISUALIZER_WINDOW_HEIGHT;;
  }
  
  SDL_SetWindowSize(visualizer->m_window, visualizer->window_width, visualizer->window_height);

  
  visualizer->m_window_renderer = SDL_CreateRenderer(visualizer->m_window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

  if(visualizer->m_window_renderer == NULL)
  {
      printf("Failed to create renderer\n");
      printf("SDL2 Error: %s\n", SDL_GetError());
      return 1;
  }

  init_ttf();
  
  pthread_t eventListenerTID;
  int rv = pthread_create(&eventListenerTID, NULL, eventListenerThreadHandler, visualizer);
  if (rv != 0) {
      printf("pthread_create failed\n");
      exit(1);
  }

  return 0;
}

void Visualizer_Coordinates_Unit2Absolute(Visualizer_t* visualizer, float xUnit_in, float yUnit_in, int32_t* xAbsolute_out, int32_t* yAbsolute_out) {
    Visualizer_AssertSanity(visualizer);
    *xAbsolute_out = (int32_t)round(xUnit_in * visualizer->window_width);
    *yAbsolute_out = (int32_t)round(yUnit_in * visualizer->window_height);
}

unsigned int Visualizer_GetWindowWidth(Visualizer_t* visualizer) {
    Visualizer_AssertSanity(visualizer);
    return visualizer->window_width;
}

unsigned int Visualizer_GetWindowHeight(Visualizer_t* visualizer) {
    Visualizer_AssertSanity(visualizer);
    return visualizer->window_height;
}

void Visualizer_Render(Visualizer_t* visualizer) {
    Visualizer_AssertSanity(visualizer);
    SDL_RenderPresent(visualizer->m_window_renderer);
}

void Visualizer_Delay(Visualizer_t* visualizer, uint32_t ms) {
    Visualizer_AssertSanity(visualizer);
    SDL_Delay(ms);
}