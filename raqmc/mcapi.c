#include "mcapi.h"
#include <yyjson.h>
#include <bcrypt/bcrypt.h>
#include <assert.h>
#include "config.h"
#include "log.h"
#include "rscall.h"
#include "measurements.h"

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

// curl -X POST -d '{"epc": "<epc", "apwd": "<access password>", "kpwd": "<kill password>", "btoken": "<bearer token>""}' http://localhost:7891/api/ite
static int __mcapi_endpoint_ite_post(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, mcapi_t* pMcapi) {
  assert(pH2oHandler != NULL);
  assert(pReq != NULL);
  assert(pMcapi != NULL);
  yyjson_doc* pJson = yyjson_read(pReq->entity.base, pReq->entity.len, 0);
  if (pJson == NULL) {
    return __mcapi_endpoint_error(pReq, 400, "Bad Request", "Invalid JSON");
  }
  yyjson_val* pRoot = yyjson_doc_get_root(pJson);
  if (pRoot == NULL || !yyjson_is_obj(pRoot)) {
      yyjson_doc_free(pJson);
      return __mcapi_endpoint_error(pReq, 400, "Bad Request", "Missing JSON root object");
  }
  yyjson_val* pEpc = yyjson_obj_get(pRoot, "epc");
  if (pEpc == NULL || !yyjson_is_str(pEpc)) {
      yyjson_doc_free(pJson);
      return __mcapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid EPC");
  }
  yyjson_val* pApwd = yyjson_obj_get(pRoot, "apwd");
  if (pApwd == NULL || !yyjson_is_str(pApwd)) {
      yyjson_doc_free(pJson);
      return __mcapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid access password");
  }
  yyjson_val* pKpwd = yyjson_obj_get(pRoot, "kpwd");
  if (pKpwd == NULL || !yyjson_is_str(pKpwd)) {
      yyjson_doc_free(pJson);
      return __mcapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid kill password");
  }
  yyjson_val* pBtoken = yyjson_obj_get(pRoot, "btoken");
  if (pBtoken == NULL || !yyjson_is_str(pBtoken)) {
      yyjson_doc_free(pJson);
      return __mcapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid bearer token");
  }

  const char* epc = yyjson_get_str(pEpc);
  const char* apwd = yyjson_get_str(pApwd);
  const char* kpwd = yyjson_get_str(pKpwd);
  const char* btoken = yyjson_get_str(pBtoken);

  assert(epc != NULL && apwd != NULL && kpwd != NULL && btoken != NULL);

  // Check bearer token //<<<<
  // if (strcmp(btoken, RAQMC_SERVER_PRE_SHARED_BEARER_TOKEN) != 0) {
  //   yyjson_doc_free(pJson);
  //   return __mcapi_endpoint_error(pReq, 401, "Unauthorized", "Invalid bearer token");
  // }
  char lkeyHashStored[BCRYPT_HASHSIZE];
  char lkeySaltStored[BCRYPT_HASHSIZE];
  strcpy(lkeyHashStored, RAQMC_LKEY_HASH);
  strcpy(lkeySaltStored, RAQMC_LKEY_SALT);

  char serverProvidedKeyHash[BCRYPT_HASHSIZE];
  assert(strlen(lkeyHashStored) == BCRYPT_HASHSIZE - 4);
  assert(strlen(lkeySaltStored) == (BCRYPT_HASHSIZE - 4)/2 - 1);
  assert(0 == bcrypt_hashpw(btoken, lkeySaltStored, serverProvidedKeyHash));
  assert(serverProvidedKeyHash[BCRYPT_HASHSIZE - 4] == '\0');
  assert(strlen(serverProvidedKeyHash) == BCRYPT_HASHSIZE - 4);
  LOG_V("__mcapi_endpoint_ite_post: serverProvidedKeyHash: %s, RAQMC_LKEY_HASH: %s, RAQMC_LKEY_SALT: %s", serverProvidedKeyHash, RAQMC_LKEY_HASH, RAQMC_LKEY_SALT);
  if (0 != strcmp(serverProvidedKeyHash, lkeyHashStored)) {
    yyjson_doc_free(pJson);
    return __mcapi_endpoint_error(pReq, 401, "Unauthorized", "Invalid bearer token");
  }

  // Create and embody the item
  const char* iePath = rscall_ie_dir_create();
  if (iePath == NULL) {
    yyjson_doc_free(pJson);
    return __mcapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to create inventory element directory");
  }
  if (0 != rscall_ie_set_epc(iePath, epc)) {
    yyjson_doc_free(pJson);
    return __mcapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to set EPC");
  }
  if (0 != rscall_ie_set_access_passwd(iePath, apwd)) {
    yyjson_doc_free(pJson);
    return __mcapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to set access password");
  }
  if (0 != rscall_ie_set_kill_passwd(iePath, kpwd)) {
    yyjson_doc_free(pJson);
    return __mcapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to set kill password");
  }
  if (0 != rscall_ie_set_flags(iePath, "00")) {
    yyjson_doc_free(pJson);
    return __mcapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to set flags");
  }
  if (0 != rscall_ie_drv_embody(iePath)) {
    yyjson_doc_free(pJson);
    return __mcapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to embody inventory element"); // TODO: Consider using 503 Service Unavailable ?
  }

  static h2o_generator_t generator = {NULL, NULL}; // TODO should we really have it static?
  pReq->res.status = 200;
  pReq->res.reason = "OK";
  h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
  h2o_start_response(pReq, &generator);

  const char* status = "success";
  const char* message = "Inventory element embodied successfully";

  // create json response
  yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
  yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
  yyjson_mut_doc_set_root(pJsonResp, pRootResp);
  yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
  yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
  // add some data as sub-object
  yyjson_mut_val* pItem = yyjson_mut_obj(pJsonResp);
  yyjson_mut_obj_add_str(pJsonResp, pItem, "epc", epc);
  // add item object to root
  yyjson_mut_obj_add_val(pJsonResp, pRootResp, "item", pItem);

  const char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
  assert(respText != NULL);
  h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
  h2o_send(pReq, &body, 1, 1);

  free((void*)respText);
  yyjson_doc_free(pJson);
  yyjson_mut_doc_free(pJsonResp);
  return 0;
}

int mcapi_endpoint_ite(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
  assert(pH2oHandler != NULL);
  assert(pReq != NULL);
  mcapi_t* pMcapi = __mcapi_self_from_h2o_handler(pH2oHandler);
  if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("POST"))) {
    return __mcapi_endpoint_ite_post(pH2oHandler, pReq, pMcapi);
  } else {
    return __mcapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
  }
}

static int __mcapi_endpoint_itq_post(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, mcapi_t* pMcapi) {
  assert(pH2oHandler != NULL);
  assert(pReq != NULL);
  assert(pMcapi != NULL);
  // TODO Figure out what RID can be sent to qps and implement qpcall and then this function
  return __mcapi_endpoint_error(pReq, 501, "Not Implemented", "Not Implemented");
}

int mcapi_endpoint_itq(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
  assert(pH2oHandler != NULL);
  assert(pReq != NULL);
  mcapi_t* pMcapi = __mcapi_self_from_h2o_handler(pH2oHandler);
  if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("POST"))) {
    return __mcapi_endpoint_itq_post(pH2oHandler, pReq, pMcapi);
  } else {
    return __mcapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
  }
}

// curl -X POST -d '{"iei": <ieIndex>, "antno": <antno>, "txp": <txPower>, "mt": <measurementType>, "btoken": "<bearer token>"}' http://localhost:7891/api/itm
static int __mcapi_endpoint_itm_post(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, mcapi_t* pMcapi) {
  assert(pH2oHandler != NULL);
  assert(pReq != NULL);
  assert(pMcapi != NULL);
  yyjson_doc* pJson = yyjson_read(pReq->entity.base, pReq->entity.len, 0);
  if (pJson == NULL) {
    return __mcapi_endpoint_error(pReq, 400, "Bad Request", "Invalid JSON");
  }
  yyjson_val* pRoot = yyjson_doc_get_root(pJson);
  if (pRoot == NULL || !yyjson_is_obj(pRoot)) {
      yyjson_doc_free(pJson);
      return __mcapi_endpoint_error(pReq, 400, "Bad Request", "Missing JSON root object");
  }
  yyjson_val* pIei = yyjson_obj_get(pRoot, "iei");
  if (pIei == NULL || !yyjson_is_int(pIei)) {
      yyjson_doc_free(pJson);
      return __mcapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid IE index");
  }
  yyjson_val* pAntno = yyjson_obj_get(pRoot, "antno");
  if (pAntno == NULL || !yyjson_is_int(pAntno)) {
      yyjson_doc_free(pJson);
      return __mcapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid antenna number");
  }
  yyjson_val* pTxp = yyjson_obj_get(pRoot, "txp");
  if (pTxp == NULL || !yyjson_is_int(pTxp)) {
      yyjson_doc_free(pJson);
      return __mcapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid tx power");
  }
  yyjson_val* pMt = yyjson_obj_get(pRoot, "mt");
  if (pMt == NULL || !yyjson_is_int(pMt)) {
      yyjson_doc_free(pJson);
      return __mcapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid measurement type");
  }
  yyjson_val* pBtoken = yyjson_obj_get(pRoot, "btoken");
  if (pBtoken == NULL || !yyjson_is_str(pBtoken)) {
      yyjson_doc_free(pJson);
      return __mcapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid bearer token");
  }

  int iei = yyjson_get_int(pIei);
  int antno = yyjson_get_int(pAntno);
  int txp = yyjson_get_int(pTxp);
  int mt = yyjson_get_int(pMt);
  const char* btoken = yyjson_get_str(pBtoken);

  assert(btoken != NULL);

  // Check bearer token
  // if (strcmp(btoken, RAQMC_SERVER_PRE_SHARED_BEARER_TOKEN) != 0) {
  //   yyjson_doc_free(pJson);
  //   return __mcapi_endpoint_error(pReq, 401, "Unauthorized", "Invalid bearer token");
  // }

  char lkeyHashStored[BCRYPT_HASHSIZE];
  char lkeySaltStored[BCRYPT_HASHSIZE];
  strcpy(lkeyHashStored, RAQMC_LKEY_HASH);
  strcpy(lkeySaltStored, RAQMC_LKEY_SALT);

  char serverProvidedKeyHash[BCRYPT_HASHSIZE];
  assert(strlen(lkeyHashStored) == BCRYPT_HASHSIZE - 4);
  assert(strlen(lkeySaltStored) == (BCRYPT_HASHSIZE - 4)/2 - 1);
  assert(0 == bcrypt_hashpw(btoken, lkeySaltStored, serverProvidedKeyHash));
  assert(serverProvidedKeyHash[BCRYPT_HASHSIZE - 4] == '\0');
  assert(strlen(serverProvidedKeyHash) == BCRYPT_HASHSIZE - 4);
  LOG_V("__mcapi_endpoint_itm_post: serverProvidedKeyHash: %s, RAQMC_LKEY_HASH: %s, RAQMC_LKEY_SALT: %s", serverProvidedKeyHash, RAQMC_LKEY_HASH, RAQMC_LKEY_SALT);
  if (0 != strcmp(serverProvidedKeyHash, lkeyHashStored)) {
    yyjson_doc_free(pJson);
    return __mcapi_endpoint_error(pReq, 401, "Unauthorized", "Invalid bearer token");
  }

  if (iei < 0) {
    yyjson_doc_free(pJson);
    return __mcapi_endpoint_error(pReq, 400, "Bad Request", "IE index must be non-negative");
  }
  if (antno < 0) {
    yyjson_doc_free(pJson);
    return __mcapi_endpoint_error(pReq, 400, "Bad Request", "Antenna number must be non-negative");
  }
  if (txp < 0) {
    yyjson_doc_free(pJson);
    return __mcapi_endpoint_error(pReq, 400, "Bad Request", "TX power must be non-negative");
  }
  if (mt != 0 && mt != 1) {
    yyjson_doc_free(pJson);
    return __mcapi_endpoint_error(pReq, 400, "Bad Request", "Invalid measurement type provided");
  }

  // Peform measurement
  int readings[2] = {0, 0};
  switch(mt) {
    case 0:
      int rv = measurements_quick_perform(iei, antno, txp, &readings[0]);
      if (rv != 0) {
        LOG_E("__mcapi_endpoint_itm_post: measurements_quick_perform failed with rv=%d", rv);
      }
      assert (rv == 0 || rv == -1 || rv == -3 || rv == -10);
      if (rv == -1) {
        yyjson_doc_free(pJson);
        return __mcapi_endpoint_error(pReq, 404, "Not Found", "Measurement (quick) failed because specified iei does not exist");
      } else if (rv == -3) {
        yyjson_doc_free(pJson);
        return __mcapi_endpoint_error(pReq, 403, "Forbidden", "Measurement (quick) failed because specified iei couldn't be read");
      } else if (rv == -10) {
        yyjson_doc_free(pJson);
        return __mcapi_endpoint_error(pReq, 500, "Not Found", "Measurement (quick) failed because specified antno does not exist");
      } else {
        assert(rv == 0);
      }
      break;
    case 1:
      rv = measurements_dual_perform(iei, antno, txp, &readings[0], &readings[1]);
      assert (rv == 0 || rv == -1 || rv == -10);
      if (rv == -1) {
        yyjson_doc_free(pJson);
        return __mcapi_endpoint_error(pReq, 404, "Not Found", "Measurement (dual) failed because specified iei does not exist");
      } else if (rv == -10) {
        yyjson_doc_free(pJson);
        return __mcapi_endpoint_error(pReq, 404, "Not Found", "Measurement (dual) failed because specified antno does not exist");
      }
      break;
    default:
      yyjson_doc_free(pJson);
      assert(0); // Should never reach here really
      return __mcapi_endpoint_error(pReq, 500, "Internal Server Error", "Invalid measurement type");
  }

  static h2o_generator_t generator = {NULL, NULL};
  pReq->res.status = 200;
  pReq->res.reason = "OK";
  h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
  h2o_start_response(pReq, &generator);

  const char* status = "success";
  const char* message = "Measurement performed successfully";

  // create json response
  yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
  yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
  yyjson_mut_doc_set_root(pJsonResp, pRootResp);
  yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
  yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
  // add data as sub-object
  yyjson_mut_val* pMeasurement = yyjson_mut_obj(pJsonResp);
  yyjson_mut_obj_add_int(pJsonResp, pMeasurement, "iei", iei);
  yyjson_mut_obj_add_int(pJsonResp, pMeasurement, "antenna_number", antno);
  yyjson_mut_obj_add_int(pJsonResp, pMeasurement, "tx_power", txp);
  yyjson_mut_obj_add_int(pJsonResp, pMeasurement, "measurement_type", mt);
  yyjson_mut_obj_add_int(pJsonResp, pMeasurement, "rssi", readings[0]);
  if (mt == 1) {
    yyjson_mut_obj_add_int(pJsonResp, pMeasurement, "read_rate", readings[1]);
  }
  // add measurement object to root
  yyjson_mut_obj_add_val(pJsonResp, pRootResp, "measurement", pMeasurement);

  const char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
  assert(respText != NULL);
  h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
  h2o_send(pReq, &body, 1, 1);

  free((void*)respText);
  yyjson_doc_free(pJson);
  yyjson_mut_doc_free(pJsonResp);
  return 0;
}

int mcapi_endpoint_itm(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
  assert(pH2oHandler != NULL);
  assert(pReq != NULL);
  mcapi_t* pMcapi = __mcapi_self_from_h2o_handler(pH2oHandler);
  if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("POST"))) {
    return __mcapi_endpoint_itm_post(pH2oHandler, pReq, pMcapi);
  } else {
    return __mcapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
  }
}