#include <gpiod.h>
#include <stdio.h>

int main() {
  const char* libgpiod_version = gpiod_api_version();
  printf("libgpiod version: %s\n", libgpiod_version);
  return 0;
}