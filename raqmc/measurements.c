#include "measurements.h"
#include <stdlib.h>
#include <assert.h>
#include "log.h"
#include "rscall.h"
#include "acall.h"

static PMutex* __pMeasurement_global_mtx = NULL;

void measurements_global_init(void) {
  if (__pMeasurement_global_mtx == NULL) {
    __pMeasurement_global_mtx = p_mutex_new();
    if (__pMeasurement_global_mtx == NULL) {
      LOG_E("measurements_global_init: Failed to allocate memory for __pMeasurement_global_mtx");
      exit(EXIT_FAILURE);
    } else {
      LOG_V("measurements_global_init: __pMeasurement_global_mtx initialized");
    }
  } else {
    LOG_W("measurements_global_init: __pMeasurement_global_mtx already initialized");
  }
}

void measurements_global_deinit(void) {
  assert(__pMeasurement_global_mtx != NULL);
  p_mutex_free(__pMeasurement_global_mtx);
  __pMeasurement_global_mtx = NULL;
  LOG_V("measurements_global_deinit: __pMeasurement_global_mtx deinitialized");
}

int measurements_quick_perform(const int ieIndex, const int antNo, const int txPower, int* pRssi_out) {
  assert(pRssi_out != NULL);
  assert(__pMeasurement_global_mtx != NULL);
  const char* iePath = rscall_ie_get_path(ieIndex);
  if (iePath == NULL) {
    LOG_E("measurements_quick_perform: Failed to get path for ieIndex %d", ieIndex);
    exit(EXIT_FAILURE);
  }
  const char* antPath = acall_ant_get_path(antNo);
  if (antPath == NULL) {
    LOG_E("measurements_quick_perform: Failed to get path for antNo %d", antNo);
    exit(EXIT_FAILURE);
  }

  p_mutex_lock(__pMeasurement_global_mtx);

  int rv = acall_ant_set_enabled(antPath);
  if (0 != rv) {
    LOG_W("measurements_quick_perform: Failed to enable antNo %d", antNo);
    p_mutex_unlock(__pMeasurement_global_mtx);
    free((void*)iePath);
    free((void*)antPath);
    if (-1 == rv) {
      return -10; // antNo doesn't exist
    } else {
      assert(0);
      return rv;
    }
  }

  *pRssi_out = 0;
  rv = rscall_ie_drv_measure_quick(iePath, txPower);
  if (0 != rv) {
    LOG_W("measurements_quick_perform: Failed to trigger quick measurement for antNo %d (rv=%d)", antNo, rv);
    p_mutex_unlock(__pMeasurement_global_mtx);
    free((void*)iePath);
    free((void*)antPath);
    return rv;
  }
  if (0 != rscall_ie_get_rssi(iePath, pRssi_out)) {
    LOG_E("measurements_quick_perform: Failed to get RSSI for ieIndex %d, antNo %d, txPower %d", ieIndex, antNo, txPower);
    p_mutex_unlock(__pMeasurement_global_mtx);
    free((void*)iePath);
    free((void*)antPath);
    exit(EXIT_FAILURE);
  }

  p_mutex_unlock(__pMeasurement_global_mtx);
  free((void*)iePath);
  free((void*)antPath);

  LOG_V("measurements_quick_perform: RSSI for ieIndex %d, antNo %d, txPower %d: %d", ieIndex, antNo, txPower, *pRssi_out);

  return 0;
}

int measurements_dual_perform(const int ieIndex, const int antNo, const int txPower, int* pRssi_out, int* pReadRate_out) {
  assert(pRssi_out != NULL);
  assert(__pMeasurement_global_mtx != NULL);
  const char* iePath = rscall_ie_get_path(ieIndex);
  if (iePath == NULL) {
    LOG_E("measurements_dual_perform: Failed to get path for ieIndex %d", ieIndex);
    exit(EXIT_FAILURE);
  }
  const char* antPath = acall_ant_get_path(antNo);
  if (antPath == NULL) {
    LOG_E("measurements_dual_perform: Failed to get path for antNo %d", antNo);
    exit(EXIT_FAILURE);
  }

  p_mutex_lock(__pMeasurement_global_mtx);

  int rv = acall_ant_set_enabled(antPath);
  if (0 != rv) {
    LOG_W("measurements_dual_perform: Failed to enable antNo %d", antNo);
    p_mutex_unlock(__pMeasurement_global_mtx);
    free((void*)iePath);
    free((void*)antPath);
    if (-1 == rv) {
      return -10; // antNo doesn't exist
    } else {
      assert(0);
      return rv;
    }
  }

  *pRssi_out = 0;
  *pReadRate_out = 0;
  rv = rscall_ie_drv_measure_dual(iePath, txPower);
  if (0 != rv) {
    LOG_W("measurements_dual_perform: Failed to trigger quick measurement for antNo %d (rv=%d)", antNo, rv);
    p_mutex_unlock(__pMeasurement_global_mtx);
    free((void*)iePath);
    free((void*)antPath);
    return rv;
  }
  if (0 != rscall_ie_get_rssi(iePath, pRssi_out)) {
    LOG_E("measurements_quick_perform: Failed to get RSSI for ieIndex %d, antNo %d, txPower %d", ieIndex, antNo, txPower);
    p_mutex_unlock(__pMeasurement_global_mtx);
    free((void*)iePath);
    free((void*)antPath);
    exit(EXIT_FAILURE);
  }
  if (0 != rscall_ie_get_read_rate(iePath, pReadRate_out)) {
    LOG_E("measurements_quick_perform: Failed to get read rate for ieIndex %d, antNo %d, txPower %d", ieIndex, antNo, txPower);
    p_mutex_unlock(__pMeasurement_global_mtx);
    free((void*)iePath);
    free((void*)antPath);
    exit(EXIT_FAILURE);
  }

  p_mutex_unlock(__pMeasurement_global_mtx);
  free((void*)iePath);
  free((void*)antPath);

  LOG_V("measurements_quick_perform: (RSSI, read rate) for ieIndex %d, antNo %d, txPower %d: (%d, %d)", ieIndex, antNo, txPower, *pRssi_out, *pReadRate_out);

  return 0;
}

void measurements_preemption_lock(void) {
  assert(__pMeasurement_global_mtx != NULL);
  assert(TRUE == p_mutex_lock(__pMeasurement_global_mtx));
}

void measurements_preemption_unlock(void) {
  assert(__pMeasurement_global_mtx != NULL);
  assert(TRUE == p_mutex_unlock(__pMeasurement_global_mtx));
}
