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
  size_t __wt_readings_buffer_len;
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
static void walker_transmit_readings_buffered(walker_t* pWalker, const int ieIndex, const int antNo, const int txp, const int rssi, const int readRate, const int mt) {
  assert(pWalker->__wt_readings_buffer_len < __WALKER_TRANSMIT_READINGS_BUFFER_MAX_LEN);
  struct walker_transmit_readings_buffer_entry* pNewEntry = &pWalker->__wt_readings_buffer[pWalker->__wt_readings_buffer_len];
  pNewEntry->antNo = antNo;
  pNewEntry->txp = txp;
  pNewEntry->rssi = rssi;
  pNewEntry->readRate = readRate;
  pNewEntry->mt = mt;
  pNewEntry->timestamp = walker_get_timestamp_precise_now();
  pNewEntry->epc = NULL;
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

    for (size_t i = 0; i < pWalker->__wt_readings_buffer_len; i++) {
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
      // Add inventory management object to the array
      yyjson_mut_arr_add_val(pInvms, pInvm);
    }
    yyjson_mut_obj_add_val(pJson, pRoot, "invms", pInvms);

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
    
    for (size_t i = 0; i < pWalker->__wt_readings_buffer_len; i++) {
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

static void* walker_task(void* pArg) {
  LOG_I("walker_task: Started");
  walker_t* pWalker = (walker_t*)pArg;
  assert(pWalker != NULL);
  puint64 inventory_cycle_counter = 0;
  while (!walker_flags_get_should_die(pWalker)) {
    inventory_cycle_counter++;
    LOG_V("walker_task: Beginning inventory cycle %llu", inventory_cycle_counter);

    int ieIndex = 0; //in
    int antNo = 0; //in
    int txPower = 10; //in
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
          walker_transmit_readings_buffered(pWalker, ieIndex, antNo, txPower, rssi, -1, 0);
        } else if (-1 == rv) {
          should_break_inventory_looper = 1;
        } else if (-3 == rv) {
          break; // We skip this ie measurements as it's impossible to read it (e.g. not embodied)
        } else if (-10 == rv) {
          should_break_antenna_looper = 1;
        } else {
          LOG_E("walker_task: measurements_quick_perform() failed with unexpected return value: %d", rv);
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