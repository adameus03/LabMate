#ifndef MEASUREMENTS_H
#define MEASUREMENTS_H

#include <plibsys/plibsys.h>

void measurements_global_init(void);

void measurements_global_deinit(void);

void measurements_quick_perform(const int ieIndex, const int antNo, const int txPower, int* pRssi_out);

void measurements_dual_perform(const int ieIndex, const int antNo, const int txPower, int* pRssi_out, int* pReadRate_out);

#endif // MEASUREMENTS_H