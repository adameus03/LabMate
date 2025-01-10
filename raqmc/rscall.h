#ifndef RSCALL_H
#define RSCALL_H

#include <stdint.h>

/**
 * @brief Create a new RSCS FS directory for Inventory Element (item), and obtain its path
 * @note RSCALL_RSCS_MOUNT_PATH needs to be defined to construct paths correctly (config.h)
 * @note Output buffer needs to be freed with free()
 */
char* rscall_ie_dir_create(void);

int rscall_ie_set_access_passwd(const char* iePath, const char* passwd);

int rscall_ie_set_kill_passwd(const char* iePath, const char* passwd);

int rscall_ie_set_epc(const char* iePath, const char* epc);

int rscall_ie_set_flags(const char* iePath, const char* flags);

int rscall_ie_drv_embody(const char* iePath);

/**
 * @brief Triggers RSSI measurement only
 */
int rscall_ie_drv_measure_quick(const char* iePath, const int txPower);

/**
 * @brief Triggers RSSI and read rate measurement
 */
int rscall_ie_drv_measure_dual(const char* iePath, const int txPower);

/**
 * @note Measurement needs to be triggered first with `rscall_ie_drv_measure_quick` or `rscall_ie_drv_measure_dual`. Otherwise the function will return a cached value.
 */
int rscall_ie_get_rssi(const char* iePath, int* pRssi);

/**
 * @note Measurement needs to be triggered first with `rscall_ie_drv_measure_dual`. Otherwise the function will return a cached value.
 */
int rscall_ie_get_read_rate(const char* iePath, int* pReadRate);

/**
 * @warning Output buffer needs to be freed with free()
 */
const char* rscall_ie_get_path(const int index);

#endif // RSCALL_H