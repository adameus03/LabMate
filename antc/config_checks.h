#ifndef CONFIG_CHECKS_H
#define CONFIG_CHECKS_H

#include "antennactl.h"
#include "config.h"

/* antennactl.h checks */
#if !defined(ANTENNACTL_HW_ARCH)
#error "ANTENNACTL_HW_ARCH is not defined"
#endif
#if (ANTENNACTL_HW_ARCH != ANTENNACTL_HW_ARCH_HMC349_DUAL)
#error "Unsupported option for ANTENNACTL_HW_ARCH"
#endif

/* log.h checks */
#if !defined(LOG_LEVEL)
#error "LOG_LEVEL is not defined"
#endif
#if ((LOG_LEVEL < LOG_FATAL) || (LOG_LEVEL > LOG_VERBOSE))
#error "Unsupported option for LOG_LEVEL"
#endif

#endif // CONFIG_CHECKS_H