#include "walker.h"
#include <plibsys/plibsys.h>
#include <assert.h>
#include <stdlib.h>
#include <curl/curl.h>
#include <yyjson.h>
#include "config.h"
#include "measurements.h"
#include "rscall.h"
#include "log.h"

#define WALKER_TRANSMIT_READINGS_ENDPOINT_URL RAQMC_LABSERV_HOST "/api/invm"
#define WALKER_TRANSMIT_READINGS_BULK_ENDPOINT_URL RAQMC_LABSERV_HOST "/api/invm-bulk"

struct walker_transmit_readings_buffer_entry {
  int antNo;
  int txp;
  int rssi;
  int readRate;
  int mt;
  char* timestamp;
  char* epc;
  int isSentry;
};

#define __WALKER_TRANSMIT_READINGS_BUFFER_MAX_LEN 32

struct walker {
  PUThread* pWalkerThread;
  //CURL* pCurl;
  CURLM* pCurlMulti;

  //int antTab[WALKER_MAX_ANTENNAS]; //what if foreign lab's aids get here?
  //size_t antTabLen;
  puint8 flags;
  struct walker_transmit_readings_buffer_entry __wt_readings_buffer[__WALKER_TRANSMIT_READINGS_BUFFER_MAX_LEN];
  int64_t __wt_readings_buffer_len;
  //int64_t __wt_sentry_ix; //start entry index
};

static int walker_flags_get_should_die(walker_t* pWalker) {
  return pWalker->flags & 0x1U;
}

static void walker_flags_set_should_die(walker_t* pWalker, int shouldDie) {
  if (shouldDie) {
    pWalker->flags |= 0x1U;
  } else {
    pWalker->flags &= ~0x1U;
  }
}

/**
 * @warning You need to free the returned buffer after use
 */
static char* walker_get_timestamp_now() {
  time_t now = time(NULL);
  struct tm* pTm = localtime(&now);
  assert(pTm != NULL);
  char* pTimestamp = (char*)malloc(20);
  assert(pTimestamp != NULL);
  assert(pTm->tm_year < 10000); //We don't want to overflow the buffer. Sorry for the Y10K bug //TODO Fix this before Y10K (:D)
  strftime(pTimestamp, 20, "%Y-%m-%d %H:%M:%S", pTm);
  LOG_V("walker_get_timestamp_now: Timestamp: %s", pTimestamp);
  return pTimestamp;
}

/**
 * @warning You need to free the returned buffer after use
 */
static char* walker_get_timestamp_precise_now() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  struct tm* pTm = localtime(&tv.tv_sec);
  assert(pTm != NULL);
  char* pTimestamp = (char*)malloc(30);
  assert(pTm->tm_year < 10000); //We don't want to overflow the buffer. Sorry for the Y10K bug //TODO Fix this before Y10K (:D)
  strftime(pTimestamp, 20, "%Y-%m-%d %H:%M:%S", pTm);
  //Add microseconds
  strcat(pTimestamp, ".");
  char micros[7];
  sprintf(micros, "%06ld", tv.tv_usec);
  strcat(pTimestamp, micros);
  LOG_V("walker_get_timestamp_precise_now: Timestamp (precise): %s", pTimestamp);
  return pTimestamp;
}

// static int __walker_curl_xferinfofn(void * p, curl_off_t o1, curl_off_t o2, curl_off_t ultotal, curl_off_t ulnow) {
//   return 1;
// }

//TODO maybe do some refactoring so that we don't mix different layers of abstraction?
static void walker_transmit_readings(walker_t* pWalker, const int ieIndex, const int antNo, const int txp, const int rssi, const int readRate, const int mt) {
  assert(pWalker != NULL);
  //assert(pWalker->pCurl != NULL);

  CURL* pCurl = curl_easy_init();
  assert(pCurl != NULL);

  // assert(CURLE_OK == curl_easy_setopt(pCurl, CURLOPT_XFERINFOFUNCTION, __walker_curl_xferinfofn));
  // assert(CURLE_OK == curl_easy_setopt(pCurl, CURLOPT_VERBOSE, 1L));
  // assert(CURLE_OK == curl_easy_setopt(pCurl, CURLOPT_FORBID_REUSE, 0L));
  // assert(CURLE_OK == curl_easy_setopt(pCurl, CURLOPT_HTTPHEADER, curl_slist_append(NULL, "Expect:")));
  
  // assert(CURLE_OK == curl_easy_setopt(pCurl, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4)); //https://stackoverflow.com/questions/48555551/using-libcurl-in-a-multithreaded-environment-causes-very-slow-performance-relate
  assert(CURLE_OK == curl_easy_setopt(pCurl, CURLOPT_CUSTOMREQUEST, "PUT")); // PUT request
  assert(CURLE_OK == curl_easy_setopt(pCurl, CURLOPT_URL, WALKER_TRANSMIT_READINGS_ENDPOINT_URL));
  yyjson_mut_doc* pJson = yyjson_mut_doc_new(NULL);
  yyjson_mut_val* pRoot = yyjson_mut_obj(pJson);
  yyjson_mut_doc_set_root(pJson, pRoot);
  char* pT = walker_get_timestamp_precise_now();
  yyjson_mut_obj_add_str(pJson, pRoot, "t", pT);
  const char* iePath = rscall_ie_get_path(ieIndex);
  char* epc = NULL;
  assert(0 == rscall_ie_get_epc(iePath, &epc));
  yyjson_mut_obj_add_str(pJson, pRoot, "epc", epc);
  //assert(pWalker->antTab != NULL);
  //assert(antNo < pWalker->antTabLen);
  //const int antId = pWalker->antTab[antNo];
  yyjson_mut_obj_add_int(pJson, pRoot, "an", antNo);
  yyjson_mut_obj_add_int(pJson, pRoot, "rxss", rssi);
  yyjson_mut_obj_add_int(pJson, pRoot, "rxrate", readRate);
  yyjson_mut_obj_add_int(pJson, pRoot, "txp", txp);
  yyjson_mut_obj_add_int(pJson, pRoot, "rxlat", -1); //TODO Implement this #w34gwfse
  yyjson_mut_obj_add_int(pJson, pRoot, "mtype", mt);
  yyjson_mut_obj_add_int(pJson, pRoot, "rkt", -1); //TODO Future #fgbefaw
  yyjson_mut_obj_add_int(pJson, pRoot, "rkp", -1); //TODO Future #awdfefe
  const char* lbToken = RAQMC_SERVER_PRE_SHARED_BEARER_TOKEN;
  yyjson_mut_obj_add_str(pJson, pRoot, "lbtoken", lbToken);

  char* jsonText = yyjson_mut_write(pJson, 0, NULL);
  assert(jsonText != NULL);
  curl_easy_setopt(pCurl, CURLOPT_POSTFIELDS, jsonText);

  LOG_V("walker_transmit_readings: Calling curl_easy_perform");
  CURLcode res = curl_easy_perform(pCurl);
  if (res != CURLE_OK) {
    LOG_E("walker_transmit_readings: curl_easy_perform() failed: %s (res=%d)", curl_easy_strerror(res), res); // TODO Should we save it and retry later? #sdvohulcisa
  } else {
    LOG_V("walker_transmit_readings: curl_easy_perform() succeeded");
  }

  //TODO Handle response if needed?
  free(jsonText);
  free(pT);
  free((void*)iePath);
  free(epc);
  yyjson_mut_doc_free(pJson);

  curl_easy_cleanup(pCurl);
}

static size_t walker_libcurl_write_data_null(void *buffer, size_t size, size_t nmemb, void *userp)
{
   return size * nmemb;
}

//TODO use async curl (multi interface) - then we need to memcpy the buffer first
/**
 * @param flags - lsb: 1 when passing first entry in the cycle, 0 otherwise
 */
static void walker_transmit_readings_buffered(walker_t* pWalker, const int ieIndex, const int antNo, const int txp, const int rssi, const int readRate, const int mt, const uint32_t flags) {
  assert(pWalker->__wt_readings_buffer_len < __WALKER_TRANSMIT_READINGS_BUFFER_MAX_LEN);
  
  ///<debug>
  if (rssi > 0) {
    LOG_D("walker_transmit_readings: RSSI for ieIndex %d, antNo %d is %d", ieIndex, antNo, rssi);
    //assert(0 == ieIndex);
  }
  ///</debug>

  // if (flags & 0x1U) {
  //   pWalker->__wt_sentry_ix = pWalker->__wt_readings_buffer_len;
  // }
  
  struct walker_transmit_readings_buffer_entry* pNewEntry = &pWalker->__wt_readings_buffer[pWalker->__wt_readings_buffer_len];
  pNewEntry->antNo = antNo;
  pNewEntry->txp = txp;
  pNewEntry->rssi = rssi;
  pNewEntry->readRate = readRate;
  pNewEntry->mt = mt;
  pNewEntry->timestamp = walker_get_timestamp_precise_now();
  pNewEntry->epc = NULL;
  pNewEntry->isSentry = (int)(flags & 0x1U);
  LOG_V("walker_transmit_readings_buffered: Calling rscall_ie_get_path");
  const char* iePath = rscall_ie_get_path(ieIndex);
  LOG_V("walker_transmit_readings_buffered: Calling rscall_ie_get_epc");
  assert(0 == rscall_ie_get_epc(iePath, &pNewEntry->epc));
  assert(pNewEntry->epc != NULL);
  LOG_V("walker_transmit_readings_buffered: rscall_ie_get_epc returned: %s", pNewEntry->epc);
  pWalker->__wt_readings_buffer_len++;

  assert(pWalker->__wt_readings_buffer_len >= 1); // >=1 because we already executed pWalker->__wt_readings_buffer_len++
  assert(pWalker->__wt_readings_buffer_len <= __WALKER_TRANSMIT_READINGS_BUFFER_MAX_LEN);
  if (pWalker->__wt_readings_buffer_len == __WALKER_TRANSMIT_READINGS_BUFFER_MAX_LEN) {
    CURL* pCurl = curl_easy_init();
    assert(pCurl != NULL);

    assert(CURLE_OK == curl_easy_setopt(pCurl, CURLOPT_CUSTOMREQUEST, "PUT")); // PUT request
    assert(CURLE_OK == curl_easy_setopt(pCurl, CURLOPT_URL, WALKER_TRANSMIT_READINGS_BULK_ENDPOINT_URL));
    //assert(CURLE_OK == curl_easy_setopt(pCurl, CURLOPT_WRITEFUNCTION, walker_libcurl_write_data_null));
    
    yyjson_mut_doc* pJson = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRoot = yyjson_mut_obj(pJson);
    yyjson_mut_doc_set_root(pJson, pRoot);
    const char* lbToken = RAQMC_SERVER_PRE_SHARED_BEARER_TOKEN;
    yyjson_mut_obj_add_str(pJson, pRoot, "lbtoken", lbToken);
    yyjson_mut_obj_add_int(pJson, pRoot, "n_invms", (int64_t)pWalker->__wt_readings_buffer_len);
    yyjson_mut_val* pInvms = yyjson_mut_arr(pJson);

    for (int64_t i = 0; i < pWalker->__wt_readings_buffer_len; i++) {
      struct walker_transmit_readings_buffer_entry* pEntry = &pWalker->__wt_readings_buffer[i];
      yyjson_mut_val* pInvm = yyjson_mut_obj(pJson);
      yyjson_mut_obj_add_str(pJson, pInvm, "t", pEntry->timestamp);
      yyjson_mut_obj_add_str(pJson, pInvm, "epc", pEntry->epc);
      yyjson_mut_obj_add_int(pJson, pInvm, "an", pEntry->antNo);
      yyjson_mut_obj_add_int(pJson, pInvm, "rxss", pEntry->rssi);
      yyjson_mut_obj_add_int(pJson, pInvm, "rxrate", pEntry->readRate);
      yyjson_mut_obj_add_int(pJson, pInvm, "txp", pEntry->txp);
      yyjson_mut_obj_add_int(pJson, pInvm, "rxlat", -1); //TODO Implement this (ref @w34gwfse)
      yyjson_mut_obj_add_int(pJson, pInvm, "mtype", pEntry->mt);
      yyjson_mut_obj_add_int(pJson, pInvm, "rkt", -1); //TODO Future (ref @fgbefaw)
      yyjson_mut_obj_add_int(pJson, pInvm, "rkp", -1); //TODO Future (ref @awdfefe)

      yyjson_mut_obj_add_bool(pJson, pInvm, "is_sentry", pEntry->isSentry ? true : false);
      // Add inventory management object to the array
      yyjson_mut_arr_add_val(pInvms, pInvm);
    }
    yyjson_mut_obj_add_val(pJson, pRoot, "invms", pInvms);
    //yyjson_mut_obj_add_uint(pJson, pRoot, "sentry_ix", pWalker->__wt_sentry_ix);

    //pWalker->__wt_sentry_ix = -1;

    LOG_V("walker_transmit_readings_buffered: Calling yyjson_mut_write");
    char* jsonText = yyjson_mut_write(pJson, 0, NULL);
    assert(jsonText != NULL);
    curl_easy_setopt(pCurl, CURLOPT_COPYPOSTFIELDS, jsonText); //we use COPYPOSTFIELDS because we want to easily free jsonText in this function. The copy will be created, managed and freed by libcurl (at least as long as we use the library provided cleanup functions)
    
    // LOG_V("walker_transmit_readings_buffered: Calling curl_easy_perform");
    // CURLcode res = curl_easy_perform(pCurl);
    // if (res != CURLE_OK) {
    //   LOG_E("walker_transmit_readings_buffered: curl_easy_perform() failed: %s (res=%d)", curl_easy_strerror(res), res); // TODO Should we save it and retry later? (ref @sdvohulcisa)
    // } else {
    //   LOG_V("walker_transmit_readings_buffered: curl_easy_perform() succeeded");
    // }

    LOG_V("walker_transmit_readings_buffered: Calling curl_multi_add_handle");
    assert(CURLE_OK == curl_multi_add_handle(pWalker->pCurlMulti, pCurl));
    int runningHandles;
    LOG_V("walker_transmit_readings_buffered: Calling curl_multi_perform");
    CURLMcode mres = curl_multi_perform(pWalker->pCurlMulti, &runningHandles);
    if (mres != CURLM_OK) {
      LOG_E("walker_transmit_readings_buffered: curl_multi_perform() failed: %d", mres);
      assert(0);
    } else {
      LOG_V("walker_transmit_readings_buffered: curl_multi_perform() succeeded, runningHandles=%d", runningHandles);
    }
    CURLMsg* pMsg = NULL;
    int msgsLeft;
    LOG_V("walker_transmit_readings_buffered: Entering curl_multi_info_read loop");
    while ((pMsg = curl_multi_info_read(pWalker->pCurlMulti, &msgsLeft))) {
      if (pMsg->msg == CURLMSG_DONE) {
        CURL* pDone = pMsg->easy_handle;
        CURLcode res = pMsg->data.result;
        if (res != CURLE_OK) {
          LOG_E("walker_transmit_readings_buffered: transfer failed: %s (res=%d)", curl_easy_strerror(res), res);
        } else {
          LOG_V("walker_transmit_readings_buffered: transfer succeeded");
        }
        curl_multi_remove_handle(pWalker->pCurlMulti, pDone);
        curl_easy_cleanup(pDone);
      }
    }
    
    for (int64_t i = 0; i < pWalker->__wt_readings_buffer_len; i++) {
      struct walker_transmit_readings_buffer_entry* pEntry = &pWalker->__wt_readings_buffer[i];
      free(pEntry->timestamp);
      free(pEntry->epc);
    }
    
    LOG_V("walker_transmit_readings_buffered: Doing json-related cleanup");
    free(jsonText);
    yyjson_mut_doc_free(pJson);
    //curl_easy_cleanup(pCurl);

    pWalker->__wt_readings_buffer_len = 0;
  }
  free((void*)iePath);
}

#define __WALKER_LIBCURL_WRITE_DATA_CAPACITY_INITIAL 1024
typedef struct __walker_libcurl_write_data {
  char* data;
  size_t pos;
  size_t capacity;
  size_t realloc_counter;
} __walker_libcurl_write_data_t;

static size_t __walker_libcurl_write(void* ptr, size_t size, size_t nmemb, void* userdata) {
  assert(ptr != NULL);
  assert(size > 0);
  assert(nmemb > 0);
  assert(userdata != NULL);
  __walker_libcurl_write_data_t* pWriteData = (__walker_libcurl_write_data_t*)userdata;
  size_t nWrite = size * nmemb;
  if (nWrite + pWriteData->pos > pWriteData->capacity) {
    //increase capacity
    size_t newCapacity = nWrite + pWriteData->pos + __WALKER_LIBCURL_WRITE_DATA_CAPACITY_INITIAL * (1 << pWriteData->realloc_counter); //additional space to reduce reallocations
    pWriteData->realloc_counter++;
    LOG_D("__walker_libcurl_write: Reallocating data buffer to %d bytes", newCapacity);
    char* newData = (char*)realloc(pWriteData->data, newCapacity);
    if (newData == NULL) {
      LOG_E("__walker_libcurl_write: realloc failed");
      assert(0);
    }
    pWriteData->data = newData;
    LOG_V("__walker_libcurl_write: Done reallocating data buffer to %d bytes", newCapacity);
    for (size_t i = pWriteData->capacity; i < newCapacity; i++) {
      pWriteData->data[i] = 0;
    }
    pWriteData->capacity = newCapacity;
  }
  memcpy(pWriteData->data + pWriteData->pos, ptr, nWrite);
  pWriteData->pos += nWrite;
  return nWrite;
}

static int walker_load_inventory(walker_t* pWalker, size_t* nItems_out) {
  assert(pWalker != NULL);

  //Construct request to labserv
  yyjson_mut_doc* pJsonReq = yyjson_mut_doc_new(NULL);
  yyjson_mut_val* pRootReq = yyjson_mut_obj(pJsonReq);
  yyjson_mut_doc_set_root(pJsonReq, pRootReq);
  const char* host = RAQMC_HOST_URL;
  const char* lbToken = RAQMC_SERVER_PRE_SHARED_BEARER_TOKEN;
  yyjson_mut_obj_add_str(pJsonReq, pRootReq, "host", host);
  yyjson_mut_obj_add_str(pJsonReq, pRootReq, "lbtoken", lbToken);
  char* reqText = yyjson_mut_write(pJsonReq, 0, NULL);
  yyjson_mut_doc_free(pJsonReq);
  if (reqText == NULL) {
    LOG_E("walker_load_inventory: yyjson_mut_write failed");
    assert(0);
  }

  //Send request to labserv
  CURL* pCurl = curl_easy_init();
  assert(pCurl != NULL);
  assert(CURLE_OK == curl_easy_setopt(pCurl, CURLOPT_URL, RAQMC_LABSERV_HOST "/api/inven-ld"));
  assert(CURLE_OK == curl_easy_setopt(pCurl, CURLOPT_POSTFIELDS, reqText));
  __walker_libcurl_write_data_t writeData = {0};
  writeData.capacity = __WALKER_LIBCURL_WRITE_DATA_CAPACITY_INITIAL;
  writeData.pos = 0;
  writeData.realloc_counter = 0;
  writeData.data = (char*)malloc(writeData.capacity);
  assert(writeData.data != NULL);
  for (int i = 0; i < writeData.capacity; i++) {
    writeData.data[i] = 0;
  }
  assert(CURLE_OK == curl_easy_setopt(pCurl, CURLOPT_WRITEFUNCTION, __walker_libcurl_write));
  assert(CURLE_OK == curl_easy_setopt(pCurl, CURLOPT_WRITEDATA, &writeData));
  CURLcode res = curl_easy_perform(pCurl);
  if (res != CURLE_OK) {
    LOG_E("walker_load_inventory: curl_easy_perform() failed: %s (res=%d)", curl_easy_strerror(res), res);
    free(reqText);
    free(writeData.data);
    curl_easy_cleanup(pCurl);
    return -1;
  }
  if (writeData.pos == 0) {
    LOG_E("walker_load_inventory: No response received");
    free(reqText);
    free(writeData.data);
    curl_easy_cleanup(pCurl);
    return -2;
  }

  //Parse response
  yyjson_doc* pJsonResp = yyjson_read(writeData.data, writeData.pos, 0);
  if (pJsonResp == NULL) {
    LOG_E("walker_load_inventory: yyjson_read failed");
    free(reqText);
    free(writeData.data);
    curl_easy_cleanup(pCurl);
    return -3;
  }
  yyjson_val* pRootResp = yyjson_doc_get_root(pJsonResp);
  if (pRootResp == NULL) {
    LOG_E("walker_load_inventory: yyjson_doc_get_root failed");
    free(reqText);
    free(writeData.data);
    curl_easy_cleanup(pCurl);
    return -4;
  }

  //parse message
  yyjson_val* pMessage = yyjson_obj_get(pRootResp, "message");
  if (pMessage == NULL || !yyjson_is_str(pMessage)) {
    LOG_E("walker_load_inventory: Failed to obtain message from json response");
    free(reqText);
    free(writeData.data);
    curl_easy_cleanup(pCurl);
    return -16;
  }
  LOG_I("walker_load_inventory: Message from labserv: %s", yyjson_get_str(pMessage));

  //parse status
  yyjson_val* pStatus = yyjson_obj_get(pRootResp, "status");
  if (pStatus == NULL || !yyjson_is_str(pStatus)) {
    LOG_E("walker_load_inventory: Failed to obtain status from json response");
    free(reqText);
    free(writeData.data);
    curl_easy_cleanup(pCurl);
    return -17;
  }
  const char* status = yyjson_get_str(pStatus);
  if (strcmp(status, "success") != 0) {
    LOG_E("walker_load_inventory: Status is not \"success\", but \"%s\"", status);
    free(reqText);
    free(writeData.data);
    curl_easy_cleanup(pCurl);
    return -18;
  }

  //parse inventory array
  yyjson_val* pInventory = yyjson_obj_get(pRootResp, "inventory");
  if (pInventory == NULL) {
    LOG_E("walker_load_inventory: Failed to obtain inventory array from json response");
    free(reqText);
    free(writeData.data);
    curl_easy_cleanup(pCurl);
    return -5;
  }
  if (!yyjson_is_arr(pInventory)) {
    LOG_E("walker_load_inventory: Inventory is not an array");
    free(reqText);
    free(writeData.data);
    curl_easy_cleanup(pCurl);
    return -6;
  }
  size_t inventoryLen = yyjson_arr_size(pInventory);
  *nItems_out = inventoryLen;
  for (size_t i = 0; i < inventoryLen; i++) {
    // Each array element is expected to be of form {"epc": "<epc>", "apwd": "<apwd>", "kpwd": "<kpwd>"}
    yyjson_val* pInventoryItem = yyjson_arr_get(pInventory, i);
    if (pInventoryItem == NULL) {
      LOG_E("walker_load_inventory: Failed to obtain inventory item (index: %d) from inventory array", i);
      free(reqText);
      free(writeData.data);
      curl_easy_cleanup(pCurl);
      return -7;
    }
    if (!yyjson_is_obj(pInventoryItem)) {
      LOG_E("walker_load_inventory: Inventory item (index: %d) is not an object", i);
      free(reqText);
      free(writeData.data);
      curl_easy_cleanup(pCurl);
      return -8;
    }
    yyjson_val* pEpc = yyjson_obj_get(pInventoryItem, "epc");
    if (pEpc == NULL || !yyjson_is_str(pEpc)) {
      LOG_E("walker_load_inventory: Failed to obtain epc from inventory item (index: %d)", i);
      free(reqText);
      free(writeData.data);
      curl_easy_cleanup(pCurl);
      return -9;
    }
    yyjson_val* pApwd = yyjson_obj_get(pInventoryItem, "apwd");
    if (pApwd == NULL || !yyjson_is_str(pApwd)) {
      LOG_E("walker_load_inventory: Failed to obtain apwd from inventory item (index: %d)", i);
      free(reqText);
      free(writeData.data);
      curl_easy_cleanup(pCurl);
      return -10;
    }
    yyjson_val* pKpwd = yyjson_obj_get(pInventoryItem, "kpwd");
    if (pKpwd == NULL || !yyjson_is_str(pKpwd)) {
      LOG_E("walker_load_inventory: Failed to obtain kpwd from inventory item (index: %d)", i);
      free(reqText);
      free(writeData.data);
      curl_easy_cleanup(pCurl);
      return -11;
    }
    const char* epc = yyjson_get_str(pEpc);
    const char* apwd = yyjson_get_str(pApwd);
    const char* kpwd = yyjson_get_str(pKpwd);
    const char* flags = "02"; //embodied

    //Register via rscall
    char* iePath = rscall_ie_dir_create();
    if (iePath == NULL) {
      LOG_E("walker_load_inventory: rscall_ie_dir_create() returned NULL");
      free(reqText);
      free(writeData.data);
      curl_easy_cleanup(pCurl);
      return -12;
    }
    int rv = rscall_ie_set_epc(iePath, epc);
    if (rv != 0) {
      LOG_E("walker_load_inventory: rscall_ie_set_epc failed: %d", rv);
      free(reqText);
      free(writeData.data);
      curl_easy_cleanup(pCurl);
      free(iePath);
      return -13;
    }
    rv = rscall_ie_set_access_passwd(iePath, apwd);
    if (rv != 0) {
      LOG_E("walker_load_inventory: rscall_ie_set_access_passwd failed: %d", rv);
      free(reqText);
      free(writeData.data);
      curl_easy_cleanup(pCurl);
      free(iePath);
      return -14;
    }
    rv = rscall_ie_set_kill_passwd(iePath, kpwd);
    if (rv != 0) {
      LOG_E("walker_load_inventory: rscall_ie_set_kill_passwd failed: %d", rv);
      free(reqText);
      free(writeData.data);
      curl_easy_cleanup(pCurl);
      free(iePath);
      return -15;
    }
    rv = rscall_ie_set_flags(iePath, flags);
    if (rv != 0) {
      LOG_E("walker_load_inventory: rscall_ie_set_flags failed: %d", rv);
      free(reqText);
      free(writeData.data);
      curl_easy_cleanup(pCurl);
      free(iePath);
      return -16;
    }
    free(iePath);
  }

  free(reqText);
  free(writeData.data);
  curl_easy_cleanup(pCurl);
  return 0;
}

static void* walker_task(void* pArg) {
  LOG_I("walker_task: Started");
  walker_t* pWalker = (walker_t*)pArg;
  assert(pWalker != NULL);

  // Load inventory
  LOG_I("walker_task: Loading inventory");
  size_t nLoadedItems = 0;
  int rv = walker_load_inventory(pWalker, &nLoadedItems);
  if (rv != 0) {
    LOG_E("walker_task: walker_load_inventory failed: rv=%d, nLoadedItems=%d", rv, nLoadedItems);
    assert(0);
  }
  LOG_I("walker_task: Inventory loaded (%d items), proceeding to walker inventory cycles", nLoadedItems);

  puint64 inventory_cycle_counter = 0;
  while (!walker_flags_get_should_die(pWalker)) {
    inventory_cycle_counter++;
    LOG_V("walker_task: Beginning inventory cycle %llu", inventory_cycle_counter);

    int ieIndex = 0; //in
    int antNo = 0; //in
    //int txPower = 10; //in
    int txPower = 24; //in
    int rssi = 0; //out
    int readRate = 0; //out

    while (!walker_flags_get_should_die(pWalker)) { //loop over inventory
      int should_break_inventory_looper = 0;
      antNo = 0;
      while (!walker_flags_get_should_die(pWalker)) { //loop over antennas
        // if (antNo >= pWalker->antTabLen) {
        //   break;
        // }
        int should_break_antenna_looper = 0;
        int rv;
        LOG_V("walker_task: Calling measurements_quick_perform");
        rv = measurements_quick_perform(ieIndex, antNo, txPower, &rssi);
        if (0 == rv) {
          LOG_V("walker_task: measurements_quick_perform() succeeded: ieIndex %d, antNo %d, txPower %d, rssi %d", ieIndex, antNo, txPower, rssi);
          //walker_transmit_readings(pWalker, ieIndex, antNo, txPower, rssi, -1, 0);
          walker_transmit_readings_buffered(pWalker, ieIndex, antNo, txPower, rssi, -1, 0, (ieIndex == 0 && antNo == 0) ? 1U : 0U);
        } else if (-1 == rv) {
          should_break_inventory_looper = 1;
        } else if (-3 == rv) {
          break; // We skip this ie measurements as it's impossible to read it (e.g. not embodied)
        } else if (-10 == rv) {
          should_break_antenna_looper = 1;
        } else {
          LOG_E("walker_task: measurements_quick_perform() failed with unexpected return value: %d (ieIndex %d, antNo %d)", rv, ieIndex, antNo);
          assert(0);
        }

        // LOG_V("walker_task: Calling measurements_dual_perform");
        // rv = measurements_dual_perform(ieIndex, antNo, txPower, &rssi, &readRate);
        // if (0 == rv) {
        //   LOG_V("walker_task: measurements_dual_perform() succeeded: ieIndex %d, antNo %d, txPower %d, rssi %d, readRate %d", ieIndex, antNo, txPower, rssi, readRate);
        //   //walker_transmit_readings(pWalker, ieIndex, antNo, txPower, rssi, readRate, 1);
        //   //assert(should_break_antenna_looper == 0);
        //   //assert(should_break_inventory_looper == 0);
        // } else if (-1 == rv) {
        //   //assert(should_break_inventory_looper == 1);
        //   should_break_inventory_looper = 1;
        // } else if (-3 == rv) {
        //   //assert(0);
        //   break; // We skip this ie measurements as it's impossible to read it (e.g. not embodied)
        // } else if (-10 == rv) {
        //   //assert(should_break_antenna_looper == 1);
        //   should_break_antenna_looper = 1;
        // } else {
        //   LOG_E("walker_task: measurements_dual_perform() failed with unexpected return value: %d", rv);
        //   assert(0);
        // }

        if (should_break_antenna_looper) {
          break;
        }
        if (should_break_inventory_looper) {
          break;
        }
        antNo++;
      }
      if (should_break_inventory_looper) {
        break;
      }
      ieIndex++;
    }
  }
  LOG_I("walker_task: Stopped");
  return NULL;
}

// void walker_init_antenna_table(walker_t* pWalker) {
//   assert(0);
// }

walker_t* walker_start_thread(void) {
  walker_t* pWalker = (walker_t*)malloc(sizeof(walker_t));
  assert(pWalker != NULL);
  //pWalker->pCurl = curl_easy_init();
  //assert(pWalker->pCurl != NULL);
  pWalker->pCurlMulti = curl_multi_init();
  assert(pWalker->pCurlMulti != NULL);

  pWalker->flags = (puint8)0U;
  pWalker->__wt_readings_buffer_len = 0;
  pWalker->pWalkerThread = p_uthread_create(walker_task, (void*)pWalker, TRUE, "walker_task");
  assert(pWalker->pWalkerThread != NULL);
  return pWalker;
}

void walker_stop_thread(walker_t* pWalker) {
  assert(pWalker != NULL);
  walker_flags_set_should_die(pWalker, TRUE);
  p_uthread_join(pWalker->pWalkerThread);
  p_uthread_unref(pWalker->pWalkerThread);
}

void walker_free_resources(walker_t* pWalker) {
  assert(pWalker != NULL);
  //curl_easy_cleanup(pWalker->pCurl);
  //pWalker->pCurl = NULL;
  curl_multi_cleanup(pWalker->pCurlMulti);
  pWalker->pCurlMulti = NULL;

  pWalker->pWalkerThread = NULL;
  pWalker->flags = (puint8)0U;
  // assert(pWalker->antTab != NULL);
  // for (size_t i = 0; i < pWalker->antTabLen; i++) {
  //   assert(pWalker->antTab[i] != NULL);
  //   free(pWalker->antTab[i]);
  // }
  free(pWalker);
}
