#include "oph.h"
#include <curl/curl.h>
#include <stdlib.h>
#include <yyjson.h>
#include <assert.h>
#include "log.h"

struct oph {
  CURL* pCurl;
  const char* host;
  const char* btoken;
};

oph_t* oph_create(const char* host, const char* btoken) {
  assert(host != NULL);
  assert(btoken != NULL);
  oph_t* pOph = (oph_t*)malloc(sizeof(oph_t));
  if (pOph == NULL) {
    return NULL;
  }
  pOph->pCurl = curl_easy_init();
  assert(pOph->pCurl != NULL);
  pOph->host = host;
  pOph->btoken = btoken;
  return pOph;
}

void oph_destroy(oph_t* pOph) {
  assert(pOph != NULL);
  assert(pOph->pCurl != NULL);
  curl_easy_cleanup(pOph->pCurl);
  pOph->pCurl = NULL;
  free(pOph);
}

#define __OPH_LIBCURL_WRITE_DATA_MAX 4096
typedef struct __oph_libcurl_write_data {
  char data[__OPH_LIBCURL_WRITE_DATA_MAX];
  size_t pos; //initialize to 0
} __oph_libcurl_write_data_t;

// Writes data to a content buffer
static size_t __oph_libcurl_write(void* ptr, size_t size, size_t nmemb, void* userdata) {
  assert(ptr != NULL);
  assert(size > 0);
  assert(nmemb > 0);
  assert(userdata != NULL);
  __oph_libcurl_write_data_t* pWriteData = (__oph_libcurl_write_data_t*)userdata;
  size_t nWrite = size * nmemb;
  if (nWrite + pWriteData->pos > __OPH_LIBCURL_WRITE_DATA_MAX) {
    LOG_E("__oph_libcurl_write: Received data exceeds buffer size (%d). Truncating to %d bytes", __OPH_LIBCURL_WRITE_DATA_MAX, pWriteData->pos);
    return 0;
  }
  memcpy(pWriteData->data + pWriteData->pos, ptr, nWrite);
  pWriteData->pos += nWrite;
  return nWrite;
}

/**
 * @warning Caller must free the returned string
 */
static char* oph_endpoint_url(oph_t* pOph, const char* endpoint) {
  assert(pOph != NULL);
  assert(pOph->host != NULL);
  assert(endpoint != NULL);
  size_t endpointStrlen = strlen(endpoint);
  assert(endpointStrlen >= 1);
  assert(endpoint[0] == '/');
  char* url = (char*)malloc(strlen(pOph->host) + endpointStrlen + 1);
  if (url == NULL) {
    return NULL;
  }
  strcpy(url, pOph->host);
  strcat(url, endpoint);
  return url;
}



int oph_trigger_embodiment(oph_t* pOph, const char* epc, const char* apwd, const char* kpwd) {
  assert(pOph != NULL);
  assert(pOph->pCurl != NULL);
  assert(pOph->host != NULL);
  assert(pOph->btoken != NULL);
  assert(epc != NULL);
  assert(apwd != NULL);
  assert(kpwd != NULL);

  //Construct request to on-premise server
  yyjson_mut_doc* pJsonReq = yyjson_mut_doc_new(NULL);
  yyjson_mut_val* pRootReq = yyjson_mut_obj(pJsonReq);
  yyjson_mut_doc_set_root(pJsonReq, pRootReq);
  yyjson_mut_obj_add_str(pJsonReq, pRootReq, "epc", epc);
  yyjson_mut_obj_add_str(pJsonReq, pRootReq, "apwd", apwd);
  yyjson_mut_obj_add_str(pJsonReq, pRootReq, "kpwd", kpwd);
  yyjson_mut_obj_add_str(pJsonReq, pRootReq, "btoken", pOph->btoken);
  char* reqText = yyjson_mut_write(pJsonReq, 0, NULL);
  yyjson_mut_doc_free(pJsonReq);
  if (reqText == NULL) {
    return -1;
  }

  //Send request to on-premise server
  const char* endpoint_url = oph_endpoint_url(pOph, "/api/ite");
  assert(CURLE_OK == curl_easy_setopt(pOph->pCurl, CURLOPT_URL, endpoint_url));
  assert(CURLE_OK == curl_easy_setopt(pOph->pCurl, CURLOPT_POSTFIELDS, reqText));
  __oph_libcurl_write_data_t writeData = {0};
  for (int i = 0; i < __OPH_LIBCURL_WRITE_DATA_MAX; i++) {
    assert(writeData.data[i] == 0);
  }
  assert(writeData.pos == 0);
  assert(CURLE_OK == curl_easy_setopt(pOph->pCurl, CURLOPT_WRITEFUNCTION, __oph_libcurl_write));
  assert(CURLE_OK == curl_easy_setopt(pOph->pCurl, CURLOPT_WRITEDATA, &writeData));
  CURLcode res = curl_easy_perform(pOph->pCurl);
  if (res != CURLE_OK) {
    LOG_E("oph_trigger_embodiment: curl_easy_perform() failed: %s (res=%d)", curl_easy_strerror(res), res);
    free(reqText);
    free((void*)endpoint_url);
    return -2;
  }

  if (writeData.pos == 0) {
    LOG_E("oph_trigger_embodiment: No response received");
    free(reqText);
    free((void*)endpoint_url);
    return -3;
  }

  //Parse response
  yyjson_doc* pJsonResp = yyjson_read(writeData.data, writeData.pos, 0);
  if (pJsonResp == NULL) {
    LOG_E("oph_trigger_embodiment: Invalid JSON response");
    free(reqText);
    free((void*)endpoint_url);
    return -4;
  }
  yyjson_val* pRootResp = yyjson_doc_get_root(pJsonResp);
  if (pRootResp == NULL || !yyjson_is_obj(pRootResp)) {
    LOG_E("oph_trigger_embodiment: Missing JSON root object in response");
    yyjson_doc_free(pJsonResp);
    free(reqText);
    free((void*)endpoint_url);
    return -5;
  }
  yyjson_val* pStatus = yyjson_obj_get(pRootResp, "status");
  if (pStatus == NULL || !yyjson_is_str(pStatus)) {
    LOG_E("oph_trigger_embodiment: Missing or invalid status in response");
    yyjson_doc_free(pJsonResp);
    free(reqText);
    free((void*)endpoint_url);
    return -6;
  }
  yyjson_val* pMessage = yyjson_obj_get(pRootResp, "message");
  if (pMessage == NULL || !yyjson_is_str(pMessage)) {
    LOG_E("oph_trigger_embodiment: Missing or invalid message in response");
    yyjson_doc_free(pJsonResp);
    free(reqText);
    free((void*)endpoint_url);
    return -7;
  }
  const char* status = yyjson_get_str(pStatus);
  const char* message = yyjson_get_str(pMessage);
  LOG_I("oph_trigger_embodiment: status=%s, message=%s", status, message);
  if (0 != strcmp(status, "success")) {
    LOG_E("oph_trigger_embodiment: status is not success");
    yyjson_doc_free(pJsonResp);
    free(reqText);
    free((void*)endpoint_url);
    return -8;
  }
  yyjson_doc_free(pJsonResp);
  free(reqText);
  free((void*)endpoint_url);
  return 0;
}

int oph_trigger_print(oph_t* pOph, void* pRfu) {
  LOG_E("oph_trigger_print: Not implemented");
  exit(EXIT_FAILURE);
  return 0;
}

int oph_trigger_measurement(oph_t* pOph, const int iei, const int antno, const int txp, const int mt) {
  LOG_E("oph_trigger_measurement: Not implemented");
  exit(EXIT_FAILURE);
  return 0;
}