#include "walker.h"
#include <plibsys/plibsys.h>
#include <assert.h>
#include <stdlib.h>
#include <curl/curl.h>
#include <yyjson.h>
#include "config.h"
#include "measurements.h"
#include "rscall.h"

#define WALKER_TRANSMIT_READINGS_ENDPOINT_URL RAQMC_LABSERV_HOST "/api/invm"

struct walker {
  PUThread* pWalkerThread;
  CURL* pCurl;
  //int antTab[WALKER_MAX_ANTENNAS]; //what if foreign lab's aids get here?
  //size_t antTabLen;
  puint8 flags;
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
  return pTimestamp;
}

//TODO maybe do some refactoring so that we don't mix different layers of abstraction?
static void walker_transmit_readings(walker_t* pWalker, const int ieIndex, const int antNo, const int txp, const int rssi, const int readRate, const int mt) {
  assert(pWalker != NULL);
  assert(pWalker->pCurl != NULL);

  assert(CURLE_OK == curl_easy_setopt(pWalker->pCurl, CURLOPT_CUSTOMREQUEST, "PUT")); // PUT request
  assert(CURLE_OK == curl_easy_setopt(pWalker->pCurl, CURLOPT_URL, WALKER_TRANSMIT_READINGS_ENDPOINT_URL));
  yyjson_mut_doc* pJson = yyjson_mut_doc_new(NULL);
  yyjson_mut_val* pRoot = yyjson_mut_obj(pJson);
  yyjson_mut_doc_set_root(pJson, pRoot);
  char* pT = walker_get_timestamp_now();
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
  yyjson_mut_obj_add_int(pJson, pRoot, "rxlat", -1); //TODO Implement this
  yyjson_mut_obj_add_int(pJson, pRoot, "mtype", mt);
  yyjson_mut_obj_add_int(pJson, pRoot, "rkt", 0); //TODO Future
  yyjson_mut_obj_add_int(pJson, pRoot, "rkp", 0); //TODO Future
  const char* lbToken = RAQMC_SERVER_PRE_SHARED_BEARER_TOKEN;
  yyjson_mut_obj_add_str(pJson, pRoot, "lbtoken", lbToken);

  char* jsonText = yyjson_mut_write(pJson, 0, NULL);
  assert(jsonText != NULL);
  curl_easy_setopt(pWalker->pCurl, CURLOPT_POSTFIELDS, jsonText);

  CURLcode res = curl_easy_perform(pWalker->pCurl);
  if (res != CURLE_OK) {
    LOG_E("walker_transmit_readings: curl_easy_perform() failed: %s (res=%d)", curl_easy_strerror(res), res);
  }
  //TODO Handle response if needed?
  free(jsonText);
  free(pT);
  free((void*)iePath);
  free(epc);
  yyjson_mut_doc_free(pJson);
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
        int rv = measurements_quick_perform(ieIndex, antNo, txPower, &rssi);
        if (0 == rv) {
          walker_transmit_readings(pWalker, ieIndex, antNo, txPower, rssi, -1, 0);
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
        rv = measurements_dual_perform(ieIndex, antNo, txPower, &rssi, &readRate);
        if (0 == rv) {
          walker_transmit_readings(pWalker, ieIndex, antNo, txPower, rssi, readRate, 1);
          assert(should_break_antenna_looper == 0);
          assert(should_break_inventory_looper == 0);
        } else if (-1 == rv) {
          assert(should_break_inventory_looper == 1);
        } else if (-3 == rv) {
          assert(0);
        } else if (-10 == rv) {
          assert(should_break_antenna_looper == 1);
        }
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
  pWalker->pCurl = curl_easy_init();
  assert(pWalker->pCurl != NULL);
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
  curl_easy_cleanup(pWalker->pCurl);
  pWalker->pCurl = NULL;
  pWalker->pWalkerThread = NULL;
  pWalker->flags = (puint8)0U;
  // assert(pWalker->antTab != NULL);
  // for (size_t i = 0; i < pWalker->antTabLen; i++) {
  //   assert(pWalker->antTab[i] != NULL);
  //   free(pWalker->antTab[i]);
  // }
  free(pWalker);
}