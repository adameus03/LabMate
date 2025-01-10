#include "antennactl.h"
#include <stdlib.h>
#include <plibsys.h>
#include <assert.h>
#include "log.h"
#include "config.h"

#if (ANTENNACTL_HW_ARCH == ANTENNACTL_HW_ARCH_HMC349_DUAL)
#include "hmc349.h"
#endif

struct antennactl {
  antennactl_target_t target;
  struct {
    struct {
#if (ANTENNACTL_HW_ARCH == ANTENNACTL_HW_ARCH_HMC349_DUAL)
      hmc349_dev_t* pHmc349;
#endif
    } hw;
    struct {
      PMutex* pHwMutex;
    } l;
  } w;
};

void antennactl_set_target(antennactl_t* pActl, antennactl_target_t target) {
  if (!pActl) {
    LOG_E("antennactl_set_target: pActl is NULL");
    exit(EXIT_FAILURE);
  }
  assert(TRUE == p_mutex_lock(pActl->w.l.pHwMutex));
#if (ANTENNACTL_HW_ARCH == ANTENNACTL_HW_ARCH_HMC349_DUAL)
  hmc349_set_outp(pActl->w.hw.pHmc349, target);
#endif
  // antennactl_target_t new_target;
  // antennactl_get_target(pActl, &new_target);
  // if (new_target != target) {
  //   LOG_E("antennactl_set_target: Failed to set target %d", target);
  //   exit(EXIT_FAILURE);
  // }
  pActl->target = target;
  assert(TRUE == p_mutex_unlock(pActl->w.l.pHwMutex));
  LOG_D("antennactl_set_target: successfully set target %d", target);
}

void antennactl_get_target(antennactl_t* pActl, antennactl_target_t* pTarget_out) {
  if (!pActl) {
    LOG_E("antennactl_get_target: pActl is NULL");
    exit(EXIT_FAILURE);
  }
  if (pTarget_out) {
    assert(TRUE == p_mutex_lock(pActl->w.l.pHwMutex));
#if (ANTENNACTL_HW_ARCH == ANTENNACTL_HW_ARCH_HMC349_DUAL)
    hmc349_get_outp(pActl->w.hw.pHmc349, (hmc349_outp_t*)pTarget_out);
#endif
    *pTarget_out = pActl->target;
    assert(TRUE == p_mutex_unlock(pActl->w.l.pHwMutex));
  } else {
    LOG_W("antennactl_get_target: pTarget_out is NULL");
  }
}

void antennactl_init(antennactl_t* pActl) {
  if (!pActl) {
    LOG_F("antennactl_init: pActl is NULL");
    exit(EXIT_FAILURE);
  }
#if (ANTENNACTL_HW_ARCH == ANTENNACTL_HW_ARCH_HMC349_DUAL)
  pActl->w.hw.pHmc349 = hmc349_dev_new();
  hmc349_dev_init(pActl->w.hw.pHmc349);
#endif

  pActl->w.l.pHwMutex = p_mutex_new();
  assert(pActl->w.l.pHwMutex != NULL);

  antennactl_get_target(pActl, &pActl->target);
  LOG_I("antennactl_init: initialized");
}

void antennactl_deinit(antennactl_t* pActl) {
  if (!pActl) {
    LOG_F("antennactl_deinit: pActl is NULL");
    exit(EXIT_FAILURE);
  }
#if (ANTENNACTL_HW_ARCH == ANTENNACTL_HW_ARCH_HMC349_DUAL)
  hmc349_dev_deinit(pActl->w.hw.pHmc349);
  hmc349_dev_free(pActl->w.hw.pHmc349);
#endif

  if (pActl->w.l.pHwMutex) { // defensive; TODO check if this is necessary
    p_mutex_free(pActl->w.l.pHwMutex);
  } else {
    LOG_W("antennactl_deinit: pActl->w.l.pHwMutex is NULL");
  }

  LOG_I("antennactl_deinit: deinitialized");
}

antennactl_t* antennactl_new() {
  antennactl_t* pActl = (antennactl_t*)malloc(sizeof(antennactl_t));
  if (!pActl) {
    LOG_F("antennactl_new: Failed to allocate memory for antennactl");
    exit(EXIT_FAILURE);
  }
  pActl->target = ANTENNACTL_TARGET_T0;
}

void antennactl_free(antennactl_t* pActl) {
  if (!pActl) {
    LOG_W("antennactl_free: pActl is NULL");
    return;
  }
  free(pActl);
}