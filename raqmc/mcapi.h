#ifndef MCAPI_H
#define MCAPI_H

#include <h2o.h>

typedef struct mcapi mcapi_t;

/**
 * @brief Create a new MCAPI instance
 */
mcapi_t* mcapi_new();

/**
 * @brief Free resource allocation caused by `mcapi_new`
 */
void mcapi_free(mcapi_t* pMcapi);

/**
 * @brief Initialize MCAPI
 */
void mcapi_init(mcapi_t* pMcapi);

/**
 * @brief Free resource allocation caused by `mcapi_init`
 */
void mcapi_deinit(mcapi_t* pMcapi);

/**
 * Handle MCAPI Inventory Trigger Embody endpoint 
 */
int mcapi_endpoint_ite(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * Handle MCAPI Inventory Trigger QR endpoint
 */
int mcapi_endpoint_itq(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

#endif // MCAPI_H