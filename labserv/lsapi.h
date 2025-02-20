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
 * @brief Handle LSAPI email verification endpoint
 */
int lsapi_endpoint_email_verify(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI service status endpoint
 */
int lsapi_endpoint_service_status(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI session endpoint
 */
int lsapi_endpoint_session(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI websocket endpoint
 */
int lsapi_endpoint_ws(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI reagent type endpoint
 */
int lsapi_endpoint_reagtype(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI reagent types endpoint
 */
int lsapi_endpoint_reagtypes(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI reagent endpoint
 */
int lsapi_endpoint_reagent(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI reagents endpoint
 */
int lsapi_endpoint_reagents(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI vendor endpoint
 */
int lsapi_endpoint_vendor(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI vendors endpoint
 */
int lsapi_endpoint_vendors(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI faculty endpoint
 */
int lsapi_endpoint_faculty(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI faculties endpoint
 */
int lsapi_endpoint_faculties(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI lab endpoint
 */
int lsapi_endpoint_lab(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI labs endpoint
 */
int lsapi_endpoint_labs(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI inventory endpoint
 */
int lsapi_endpoint_inventory(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI inventory items endpoint
 */
int lsapi_endpoint_inventory_items(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI INVENtory LoaD endpoint
 */
int lsapi_endpoint_inven_ld(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI antenna endpoint
 */
int lsapi_endpoint_antenna(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI INVentory Measurement endpoint
 */
int lsapi_endpoint_invm(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

/**
 * @brief Handle LSAPI INVentory Measurement BULK endpoint
 */
int lsapi_endpoint_invm_bulk(h2o_handler_t* pH2oHandler, h2o_req_t* pReq);

#endif //LSAPI_H