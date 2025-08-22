#ifndef CONFIG_H
#define CONFIG_H

/* RSCALL configuration */
#define RSCALL_RSCS_MOUNT_PATH "/mnt/rscs"
#define RSCALL_IE_DRV_DUAL_MEASURE_TIMEOUT 100000 /* 100000 us */

/* ACALL configuration */
#define ACALL_ANTC_MOUNT_PATH "/mnt/antc"

/* Logging configuration */
#include "log-levels.h"
#define LOG_LEVEL LOG_VERBOSE

#endif //CONFIG_H