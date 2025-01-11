#ifndef RAQMC_CONFIG_H
#define RAQMC_CONFIG_H

#define RAQMC_USE_HTTPS 0
#define RAQMC_USE_MEMCACHED 0
#define RAQMC_IPV4_ADDR "0.0.0.0"
#define RAQMC_IP_PORT 7891
#define RAQMC_H2O_ACCESS_LOG_FILE_PATH "/dev/stdout"
//TODO Configure this token in a better place so that changing it doesn't require recompilation?
#define RAQMC_SERVER_PRE_SHARED_BEARER_TOKEN "<pre-shared-bearer-token>"
#define RAQMC_LABSERV_HOST "<labserv-host>"

/* RSCALL configuration */
#define RSCALL_RSCS_MOUNT_PATH "/mnt/rscs"
#define RSCALL_IE_DRV_DUAL_MEASURE_TIMEOUT 100000 /* 100000 us */

/* ACALL configuration */
#define ACALL_ANTC_MOUNT_PATH "/mnt/antc"

/* Logging configuration */
#include "log.h"
#define LOG_LEVEL LOG_VERBOSE

#endif // RAQMC_CONFIG_H