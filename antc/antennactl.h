#ifndef ANTENNACTL_H
#define ANTENNACTL_H

#include "config.h"

#define ANTENNACTL_HW_ARCH_HMC349_DUAL 1

typedef struct antennactl antennactl_t; // opaque type for antennactl main struct

typedef enum antennactl_target {
  ANTENNACTL_TARGET_T0,
  ANTENNACTL_TARGET_T1
} antennactl_target_t;

/**
 * @brief Set target for antennactl
 */
void antennactl_set_target(antennactl_t* pActl, antennactl_target_t target);

/**
 * @brief Get target for antennactl
 */
void antennactl_get_target(antennactl_t* pActl, antennactl_target_t* pTarget_out);

/**
 * @brief Initialize antennactl using underlying hardware driver
 */
void antennactl_init(antennactl_t* pActl);

/**
 * @brief Deinitialize antennactl using underlying hardware driver
 */
void antennactl_deinit(antennactl_t* pActl);

/**
 * @brief Allocate resources for antennactl main struct
 */
antennactl_t* antennactl_new();

/**
 * @brief Free resources allocated by `antennactl_new`
 */
void antennactl_free(antennactl_t* pActl);


#endif // ANTENNACTL_H