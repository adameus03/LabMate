/**
 * @file oph.h
 * @brief On-Premise Host (OPH) connector api
 */

#ifndef OPH_H
#define OPH_H

typedef struct oph oph_t;

/**
 * @brief Create a new OPH instance
 */
oph_t* oph_create(const char* host, const char* btoken);

/**
 * @brief Destroy an OPH instance - free resources allocated by `oph_create`
 */
void oph_destroy(oph_t* pOph);

int oph_trigger_embodiment(oph_t* pOph, const char* epc, const char* apwd, const char* kpwd);

/**
 * TODO: Implement
 */
int oph_trigger_print(oph_t* pOph, void* pRfu);

int oph_trigger_measurement(oph_t* pOph, const int iei, const int antno, const int txp, const int mt);

#endif // OPH_H