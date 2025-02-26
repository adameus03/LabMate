#include "db.h"

// Opaque type for tracker management
typedef struct tracker tracker_t;

/**
 * @brief Low-level measurement data (invm) processing function
 * @warning Should only be called with a full data cycle (i.e. beginning with a sentry record and containing ordered records up to the next sentry record exclusive)
 * @note tracker_process_data_buffered is a wrapper around tracker_process_data that buffers data until next sentry record is found - a useful frontend.
 */
int tracker_process_data(tracker_t* pTracker,
                         const int nInvm,
                         const int nCycles,
                         const char** pTimes, 
                         const char** pEpcs, 
                         const char** pAntnos, 
                         const char** pRxsss, 
                         const char** pRxrates,
                         const char** pTxps,
                         const char** pRxlats, 
                         const char** pMtypes, 
                         const char** pRkts, 
                         const char** pRkps);

/**
 * @brief Application wrapper around `tracker_process_data` that buffers data until next sentry record is found
 * @note This function is designed to be thread-safe, though it doesn't make sense to call it from multiple threads for the same lab as we block the critical section with a mutex anyway
 */
int tracker_process_data_buffered(tracker_t* pTracker, 
                                  const int nInvm,
                                  const char** pTimes, 
                                  const char** pEpcs, 
                                  const char** pAntnos, 
                                  const char** pRxsss, 
                                  const char** pRxrates,
                                  const char** pTxps,
                                  const char** pRxlats, 
                                  const char** pMtypes, 
                                  const char** pRkts, 
                                  const char** pRkps,
                                  const int* pIsSentry);

/**
 * @brief Create a new tracker instance
 * @note Resources allocation caused by this function and subsequent operations shall be freed after use by calling `tracker_free`. Usually one tracker instance per application should be enough though.
 */
tracker_t* tracker_new(db_t* pDb);

/**
 * @brief Free resource allocation caused by `tracker_new` and subsequent operations
 */
void tracker_free(tracker_t* pTracker);

