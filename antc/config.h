#ifndef CONFIG_H
#define CONFIG_H

/* GPIOD configuration for HMC349 */
#define HMC349_GPIOD_CHIP "/dev/gpiochip0"
#define HMC349_GPIOD_CHIP_LINE 17
#define HMC349_GPIOD_CHIP_LINE_DEFAULT_VAL 0

/* antennactl configuration */
#include "antennactl.h"
#define ANTENNACTL_HW_ARCH ANTENNACTL_HW_ARCH_HMC349_DUAL
//#define ANTENNACTL_HW_ARCH_MAP_T0 "0"

/* Logging configuration */
#include "log.h"
#define LOG_LEVEL LOG_VERBOSE

#endif // CONFIG_H