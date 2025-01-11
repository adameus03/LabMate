#ifndef MEASUREMENTS_H
#define MEASUREMENTS_H

#include <plibsys/plibsys.h>

void measurements_global_init(void);

void measurements_global_deinit(void);

/**
 * @return 0 on success, -1 when ieIndex doesn't exist, -10 when antNo doesn't exist, other negative values on other errors
 */
int measurements_quick_perform(const int ieIndex, const int antNo, const int txPower, int* pRssi_out);

/**
 * @return 0 on success, -1 when ieIndex doesn't exist, -10 when antNo doesn't exist, other negative values on other errors
 */
int measurements_dual_perform(const int ieIndex, const int antNo, const int txPower, int* pRssi_out, int* pReadRate_out);

#endif // MEASUREMENTS_H