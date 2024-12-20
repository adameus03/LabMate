/**
 * LSAPI - Labserv API for HTTP backend endpoints
 */

#ifndef LSAPI_H
#define LSAPI_H

#include <h2o.h>

typedef struct lsapi lsapi_t;

/**
 * @brief Create a new LSAPI instance
 */
lsapi_t* lsapi_new();

/**
 * @brief Free resource allocation caused by `lsapi_new`
 */
void lsapi_free(lsapi_t* pLsapi);

/**
 * @brief Initialize LSAPI
 */
void lsapi_init(lsapi_t* pLsapi);

/**
 * @brief Free resource allocation caused by `lsapi_init`
 */
void lsapi_deinit(lsapi_t* pLsapi);

/**
 * @brief Handle LSAPI user endpoint
 */
int lsapi_endpoint_user(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI service status endpoint
 */
int lsapi_endpoint_service_status(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

#endif //LSAPI_H