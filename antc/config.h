#ifndef CONFIG_H
#define CONFIG_H

// GPIOD configuration for HMC349
#define HMC349_GPIOD_CHIP "/dev/gpiochip0"
#define HMC349_GPIOD_CHIP_LINE 17
#define HMC349_GPIOD_CHIP_LINE_DEFAULT_VAL 0

// Logging configuration
#include "log.h"
#define LOG_LEVEL LOG_VERBOSE

#endif