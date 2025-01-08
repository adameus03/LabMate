#include "mcapi.h"
#include <yyjson.h>
#include "config.h"
#include "log.h"

struct mcapi {

};

mcapi_t* mcapi_new() {
    mcapi_t* pMcapi = (mcapi_t*)malloc(sizeof(mcapi_t));
    if (pMcapi == NULL) {
        LOG_E("mcapi_new: Failed to allocate memory for mcapi_t");
        exit(EXIT_FAILURE);
    }

    return pMcapi;
}

void mcapi_free(mcapi_t* pMcapi) {
    assert(pMcapi != NULL);
    free(pMcapi);
}

void mcapi_init(mcapi_t* pMcapi) {
    assert(pMcapi != NULL);
    //pMcapi->pDb = db_new();
    //db_init(pMcapi->pDb);
}

void mcapi_deinit(mcapi_t* pMcapi) {
    assert(pMcapi != NULL);
    //db_close(pMcapi->pDb);
    //db_free(pMcapi->pDb);
}

/**
 * @brief Trick function to obtain mcapi_t* using h2o_handler_t*
 */
static mcapi_t* __mcapi_self_from_h2o_handler(h2o_handler_t* pH2oHandler) {
    assert(pH2oHandler != NULL);
    return (mcapi_t*)*(void**)(pH2oHandler + 1);
}

static int __mcapi_endpoint_resp_short(h2o_req_t *pReq, 
                                       const int httpStatus, 
                                       const char* httpReason, 
                                       const char* jsonStatus, 
                                       const char* jsonMessage) {
    assert(pReq != NULL);
    assert(httpReason != NULL);
    assert(jsonStatus != NULL);
    assert(jsonMessage != NULL);
    static h2o_generator_t generator = {NULL, NULL};
    //pReq->res.status = httpStatus;
    pReq->res.status = 200; //TODO remove this, however using a non-200 status code for some reason delays the response and we don't know why, so it needs to be resolved first
    pReq->res.reason = httpReason;
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pJsonRespRoot = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pJsonRespRoot);

    yyjson_mut_obj_add_str(pJsonResp, pJsonRespRoot, "status", jsonStatus);
    yyjson_mut_obj_add_str(pJsonResp, pJsonRespRoot, "message", jsonMessage);
    const char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);
    free((void*)respText);
    yyjson_mut_doc_free(pJsonResp);
    return 0;
}

static int __mcapi_endpoint_error(h2o_req_t *pReq, const int status, const char* reason, const char* errMessage) {
    return __mcapi_endpoint_resp_short(pReq, status, reason, "error", errMessage);
}

static int __mcapi_endpoint_success(h2o_req_t *pReq, const int status, const char* reason, const char* message) {
    return __mcapi_endpoint_resp_short(pReq, status, reason, "success", message);
}

int mcapi_endpoint_ite(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
  
}

int mcapi_endpoint_itq(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {

}