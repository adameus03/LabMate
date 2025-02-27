#include "tracker.h"
#include <assert.h>
#include <string.h>
#include <plibsys/plibsys.h>
#include "log.h"

typedef struct tracker_lab_buffer {
  PMutex* pMutex; // don't allow multiple threads for the same lab to process data simultaneously
  int lab_id;
  int buf_len; //current buffer length
  int buf_capacity; //current buffer capacity
  char** pTimes;
  char** pEpcs;
  char** pAntnos;
  char** pRxsss;
  char** pRxrates;
  char** pTxps;
  char** pRxlats;
  char** pMtypes;
  char** pRkts;
  char** pRkps;
} tracker_lab_buffer_t;

#define TRACKER_LAB_BUFFER_ALLOCATION_UNIT 32
#define TRACKER_LAB_BUFFER_REALLOCATION_RATIO 2

static tracker_lab_buffer_t tracker_lab_buffer_new(int lab_id) {
  PMutex* pMutex = p_mutex_new();
  assert(pMutex != NULL);
  // return (tracker_lab_buffer_t) {
  //   .pMutex = pMutex,
  //   .lab_id = lab_id,
  //   .buf_len = 0,
  //   .pTimes = NULL,
  //   .pEpcs = NULL,
  //   .pAntnos = NULL,
  //   .pRxsss = NULL,
  //   .pRxrates = NULL,
  //   .pTxps = NULL,
  //   .pRxlats = NULL,
  //   .pMtypes = NULL,
  //   .pRkts = NULL,
  //   .pRkps = NULL
  // };
  tracker_lab_buffer_t labBuffer = {
    .pMutex = pMutex,
    .lab_id = lab_id,
    .buf_len = 0,
    .buf_capacity = TRACKER_LAB_BUFFER_ALLOCATION_UNIT,
    .pTimes = (char**)malloc(TRACKER_LAB_BUFFER_ALLOCATION_UNIT * sizeof(char*)),
    .pEpcs = (char**)malloc(TRACKER_LAB_BUFFER_ALLOCATION_UNIT * sizeof(char*)),
    .pAntnos = (char**)malloc(TRACKER_LAB_BUFFER_ALLOCATION_UNIT * sizeof(char*)),
    .pRxsss = (char**)malloc(TRACKER_LAB_BUFFER_ALLOCATION_UNIT * sizeof(char*)),
    .pRxrates = (char**)malloc(TRACKER_LAB_BUFFER_ALLOCATION_UNIT * sizeof(char*)),
    .pTxps = (char**)malloc(TRACKER_LAB_BUFFER_ALLOCATION_UNIT * sizeof(char*)),
    .pRxlats = (char**)malloc(TRACKER_LAB_BUFFER_ALLOCATION_UNIT * sizeof(char*)),
    .pMtypes = (char**)malloc(TRACKER_LAB_BUFFER_ALLOCATION_UNIT * sizeof(char*)),
    .pRkts = (char**)malloc(TRACKER_LAB_BUFFER_ALLOCATION_UNIT * sizeof(char*)),
    .pRkps = (char**)malloc(TRACKER_LAB_BUFFER_ALLOCATION_UNIT * sizeof(char*))
  };
  assert(labBuffer.pTimes != NULL);
  assert(labBuffer.pEpcs != NULL);
  assert(labBuffer.pAntnos != NULL);
  assert(labBuffer.pRxsss != NULL);
  assert(labBuffer.pRxrates != NULL);
  assert(labBuffer.pTxps != NULL);
  assert(labBuffer.pRxlats != NULL);
  assert(labBuffer.pMtypes != NULL);
  assert(labBuffer.pRkts != NULL);
  assert(labBuffer.pRkps != NULL);
  for (int i = 0; i < TRACKER_LAB_BUFFER_ALLOCATION_UNIT; i++) {
    labBuffer.pTimes[i] = NULL;
    labBuffer.pEpcs[i] = NULL;
    labBuffer.pAntnos[i] = NULL;
    labBuffer.pRxsss[i] = NULL;
    labBuffer.pRxrates[i] = NULL;
    labBuffer.pTxps[i] = NULL;
    labBuffer.pRxlats[i] = NULL;
    labBuffer.pMtypes[i] = NULL;
    labBuffer.pRkts[i] = NULL;
    labBuffer.pRkps[i] = NULL;
  }
  return labBuffer;
}

static void tracker_lab_buffer_expand(tracker_lab_buffer_t* pLabBuffer) {
  assert(pLabBuffer != NULL);
  int newBufCapacity = pLabBuffer->buf_capacity * TRACKER_LAB_BUFFER_REALLOCATION_RATIO;
  pLabBuffer->pTimes = (char**)realloc(pLabBuffer->pTimes, newBufCapacity * sizeof(char*));
  pLabBuffer->pEpcs = (char**)realloc(pLabBuffer->pEpcs, newBufCapacity * sizeof(char*));
  pLabBuffer->pAntnos = (char**)realloc(pLabBuffer->pAntnos, newBufCapacity * sizeof(char*));
  pLabBuffer->pRxsss = (char**)realloc(pLabBuffer->pRxsss, newBufCapacity * sizeof(char*));
  pLabBuffer->pRxrates = (char**)realloc(pLabBuffer->pRxrates, newBufCapacity * sizeof(char*));
  pLabBuffer->pTxps = (char**)realloc(pLabBuffer->pTxps, newBufCapacity * sizeof(char*));
  pLabBuffer->pRxlats = (char**)realloc(pLabBuffer->pRxlats, newBufCapacity * sizeof(char*));
  pLabBuffer->pMtypes = (char**)realloc(pLabBuffer->pMtypes, newBufCapacity * sizeof(char*));
  pLabBuffer->pRkts = (char**)realloc(pLabBuffer->pRkts, newBufCapacity * sizeof(char*));
  pLabBuffer->pRkps = (char**)realloc(pLabBuffer->pRkps, newBufCapacity * sizeof(char*));
  assert(pLabBuffer->pTimes != NULL);
  assert(pLabBuffer->pEpcs != NULL);
  assert(pLabBuffer->pAntnos != NULL);
  assert(pLabBuffer->pRxsss != NULL);
  assert(pLabBuffer->pRxrates != NULL);
  assert(pLabBuffer->pTxps != NULL);
  assert(pLabBuffer->pRxlats != NULL);
  assert(pLabBuffer->pMtypes != NULL);
  assert(pLabBuffer->pRkts != NULL);
  assert(pLabBuffer->pRkps != NULL);
  for (int i = pLabBuffer->buf_capacity; i < newBufCapacity; i++) {
    pLabBuffer->pTimes[i] = NULL;
    pLabBuffer->pEpcs[i] = NULL;
    pLabBuffer->pAntnos[i] = NULL;
    pLabBuffer->pRxsss[i] = NULL;
    pLabBuffer->pRxrates[i] = NULL;
    pLabBuffer->pTxps[i] = NULL;
    pLabBuffer->pRxlats[i] = NULL;
    pLabBuffer->pMtypes[i] = NULL;
    pLabBuffer->pRkts[i] = NULL;
    pLabBuffer->pRkps[i] = NULL;
  }
  pLabBuffer->buf_capacity = newBufCapacity;
}

//TODO Prevent malicious labs from causing memory exhaustion by sending a huge number of invms without sentry records
static void tracker_lab_buffer_concat(tracker_lab_buffer_t* pLabBuffer, const int nInvms, const char** pTimes, const char** pEpcs, const char** pAntnos, const char** pRxsss, const char** pRxrates, const char** pTxps, const char** pRxlats, const char** pMtypes, const char** pRkts, const char** pRkps) {
  assert(pLabBuffer != NULL);
  assert(pLabBuffer->buf_len <= pLabBuffer->buf_capacity);
  int newBufLen = pLabBuffer->buf_len + nInvms;
  if (newBufLen > pLabBuffer->buf_capacity) {
    tracker_lab_buffer_expand(pLabBuffer);
  }
  for (int i = 0; i < nInvms; i++) {
    pLabBuffer->pTimes[pLabBuffer->buf_len] = p_strdup(pTimes[i]);
    pLabBuffer->pEpcs[pLabBuffer->buf_len] = p_strdup(pEpcs[i]);
    pLabBuffer->pAntnos[pLabBuffer->buf_len] = p_strdup(pAntnos[i]);
    pLabBuffer->pRxsss[pLabBuffer->buf_len] = p_strdup(pRxsss[i]);
    pLabBuffer->pRxrates[pLabBuffer->buf_len] = p_strdup(pRxrates[i]);
    pLabBuffer->pTxps[pLabBuffer->buf_len] = p_strdup(pTxps[i]);
    pLabBuffer->pRxlats[pLabBuffer->buf_len] = p_strdup(pRxlats[i]);
    pLabBuffer->pMtypes[pLabBuffer->buf_len] = p_strdup(pMtypes[i]);
    pLabBuffer->pRkts[pLabBuffer->buf_len] = p_strdup(pRkts[i]);
    pLabBuffer->pRkps[pLabBuffer->buf_len] = p_strdup(pRkps[i]);
    pLabBuffer->buf_len++;
  }
}

/**
 * @brief Free the first `nInvms` records from the buffer and shift the remaining records to the beginning while adjusting the buffer length
 */
static void tracker_lab_buffer_unbuffer(tracker_lab_buffer_t* pLabBuffer, const int nInvms) {
  assert(pLabBuffer != NULL);
  assert(pLabBuffer->buf_len >= nInvms);
  for (int i = 0; i < nInvms; i++) {
    free(pLabBuffer->pTimes[i]);
    free(pLabBuffer->pEpcs[i]);
    free(pLabBuffer->pAntnos[i]);
    free(pLabBuffer->pRxsss[i]);
    free(pLabBuffer->pRxrates[i]);
    free(pLabBuffer->pTxps[i]);
    free(pLabBuffer->pRxlats[i]);
    free(pLabBuffer->pMtypes[i]);
    free(pLabBuffer->pRkts[i]);
    free(pLabBuffer->pRkps[i]);
  }
  for (int i = nInvms; i < pLabBuffer->buf_len; i++) {
    pLabBuffer->pTimes[i - nInvms] = pLabBuffer->pTimes[i];
    pLabBuffer->pEpcs[i - nInvms] = pLabBuffer->pEpcs[i];
    pLabBuffer->pAntnos[i - nInvms] = pLabBuffer->pAntnos[i];
    pLabBuffer->pRxsss[i - nInvms] = pLabBuffer->pRxsss[i];
    pLabBuffer->pRxrates[i - nInvms] = pLabBuffer->pRxrates[i];
    pLabBuffer->pTxps[i - nInvms] = pLabBuffer->pTxps[i];
    pLabBuffer->pRxlats[i - nInvms] = pLabBuffer->pRxlats[i];
    pLabBuffer->pMtypes[i - nInvms] = pLabBuffer->pMtypes[i];
    pLabBuffer->pRkts[i - nInvms] = pLabBuffer->pRkts[i];
    pLabBuffer->pRkps[i - nInvms] = pLabBuffer->pRkps[i];
  }
  pLabBuffer->buf_len -= nInvms;
}

static void tracker_lab_buffer_clear(tracker_lab_buffer_t* pLabBuffer) {
  assert(pLabBuffer != NULL);
  for (int i = 0; i < pLabBuffer->buf_len; i++) {
    free(pLabBuffer->pTimes[i]);
    free(pLabBuffer->pEpcs[i]);
    free(pLabBuffer->pAntnos[i]);
    free(pLabBuffer->pRxsss[i]);
    free(pLabBuffer->pRxrates[i]);
    free(pLabBuffer->pTxps[i]);
    free(pLabBuffer->pRxlats[i]);
    free(pLabBuffer->pMtypes[i]);
    free(pLabBuffer->pRkts[i]);
    free(pLabBuffer->pRkps[i]);
  }
  pLabBuffer->buf_len = 0;
}

// @warning Doesn't free the provided pointer itself - only the internal stuff
static void tracker_lab_buffer_free(tracker_lab_buffer_t* pLabBuffer) {
  assert(pLabBuffer != NULL);
  assert(pLabBuffer->pMutex != NULL);
  p_mutex_free(pLabBuffer->pMutex);
  if (pLabBuffer->pTimes != NULL) {
    for (int i = 0; i < pLabBuffer->buf_len; i++) {
      if (pLabBuffer->pTimes[i] != NULL) {
        free(pLabBuffer->pTimes[i]);
      }
    }
    free(pLabBuffer->pTimes);
  }
  if (pLabBuffer->pEpcs != NULL) {
    for (int i = 0; i < pLabBuffer->buf_len; i++) {
      if (pLabBuffer->pEpcs[i] != NULL) {
        free(pLabBuffer->pEpcs[i]);
      }
    }
    free(pLabBuffer->pEpcs);
  }
  if (pLabBuffer->pAntnos != NULL) {
    for (int i = 0; i < pLabBuffer->buf_len; i++) {
      if (pLabBuffer->pAntnos[i] != NULL) {
        free(pLabBuffer->pAntnos[i]);
      }
    }
    free(pLabBuffer->pAntnos);
  }
  if (pLabBuffer->pRxsss != NULL) {
    for (int i = 0; i < pLabBuffer->buf_len; i++) {
      if (pLabBuffer->pRxsss[i] != NULL) {
        free(pLabBuffer->pRxsss[i]);
      }
    }
    free(pLabBuffer->pRxsss);
  }
  if (pLabBuffer->pRxrates != NULL) {
    for (int i = 0; i < pLabBuffer->buf_len; i++) {
      if (pLabBuffer->pRxrates[i] != NULL) {
        free(pLabBuffer->pRxrates[i]);
      }
    }
    free(pLabBuffer->pRxrates);
  }
  if (pLabBuffer->pTxps != NULL) {
    for (int i = 0; i < pLabBuffer->buf_len; i++) {
      if (pLabBuffer->pTxps[i] != NULL) {
        free(pLabBuffer->pTxps[i]);
      }
    }
    free(pLabBuffer->pTxps);
  }
  if (pLabBuffer->pRxlats != NULL) {
    for (int i = 0; i < pLabBuffer->buf_len; i++) {
      if (pLabBuffer->pRxlats[i] != NULL) {
        free(pLabBuffer->pRxlats[i]);
      }
    }
    free(pLabBuffer->pRxlats);
  }
  if (pLabBuffer->pMtypes != NULL) {
    for (int i = 0; i < pLabBuffer->buf_len; i++) {
      if (pLabBuffer->pMtypes[i] != NULL) {
        free(pLabBuffer->pMtypes[i]);
      }
    }
    free(pLabBuffer->pMtypes);
  }
  if (pLabBuffer->pRkts != NULL) {
    for (int i = 0; i < pLabBuffer->buf_len; i++) {
      if (pLabBuffer->pRkts[i] != NULL) {
        free(pLabBuffer->pRkts[i]);
      }
    }
    free(pLabBuffer->pRkts);
  }
  if (pLabBuffer->pRkps != NULL) {
    for (int i = 0; i < pLabBuffer->buf_len; i++) {
      if (pLabBuffer->pRkps[i] != NULL) {
        free(pLabBuffer->pRkps[i]);
      }
    }
    free(pLabBuffer->pRkps);
  }
}

struct tracker {
  db_t* pDb;
  tracker_lab_buffer_t* pLabBuffers;
  int nLabBuffers;
  PMutex* pMutex; // for pLabBuffers reallocations
};

static int tracker_lab_buffer_is_empty(tracker_lab_buffer_t* pLabBuffer) {
  return pLabBuffer->buf_len == 0;
}

static tracker_lab_buffer_t* tracker_lab_buffer_find(tracker_t* pTracker, int lab_id) {
  for (int i = 0; i < pTracker->nLabBuffers; i++) {
    if (pTracker->pLabBuffers[i].lab_id == lab_id) {
      return &pTracker->pLabBuffers[i];
    }
  }
  return NULL;
}

static int __tracker_get_lab(db_t* pDb, const char* epc, db_lab_t* pLab_out) {
  db_lab_t lab;
  int rv = db_lab_get_by_epc(pDb, epc, &lab);
  if (0 != rv) {
    if (rv == -2) {
      LOG_W("__tracker_get_lab: EPC %s does not match any lab", epc);
      return -1;
    } else {
      LOG_E("__tracker_get_lab: db_lab_get_by_epc failed: %d", rv);
      return -2;
    }
  }
  *pLab_out = lab;
  return 0;
}

//TODO move these tracker and lsapi functions to a separate utils file?
/**
 * @warning You need to free the returned buffer after use
 */
static char* __tracker_itoa(int n) {
  assert(sizeof(int) <= 4);
  char* buf = (char*)malloc(12); // 12 bytes is enough for 32-bit int
  if (buf == NULL) {
      return NULL;
  }
  snprintf(buf, 12, "%d", n);
  return buf;
}

/**
 * @warning You need to free the returned buffer after use
 */
static char* __tracker_dtoa(double d) {
  char* buf = (char*)malloc(32); // 32 bytes should be enough for double
  if (buf == NULL) {
      return NULL;
  }
  snprintf(buf, 32, "%f", d);
  return buf;
}

typedef struct tracker_point {
  float x;
  float y;
  float z;
} tracker_point_t;

#define TRACKER_MVECTOR_NDIMS_MAX 128
#define TRACKER_TIMESTAMP_LEN_MAX 32

typedef tracker_point_t tracker_target_location_t;

//measurement vector
typedef struct tracker_mvector {
  int rssi[TRACKER_MVECTOR_NDIMS_MAX];
  char timestamp[TRACKER_TIMESTAMP_LEN_MAX+1];
  int componentCount;
} tracker_mvector_t;

//multi-mvector
typedef struct tracker_mmvector {
  tracker_mvector_t* mvectors;
  int nMvectors;
} tracker_mmvector_t;

typedef struct tracker_physical_basepoint {
  tracker_point_t point;
  tracker_mmvector_t mmvector;
} tracker_physical_basepoint_t;

typedef struct tracker_target_timed_location {
  tracker_target_location_t location;
  char timestamp[TRACKER_TIMESTAMP_LEN_MAX+1];
  tracker_mmvector_t* pIob; // corresponding input mmvector
} tracker_target_timed_location_t;

#define TRACKER_TARGET_TIMED_LOCATION_ARRAY_MAXLEN 256

typedef struct tracker_target_timed_location_array {
  tracker_target_timed_location_t pLocations[TRACKER_TARGET_TIMED_LOCATION_ARRAY_MAXLEN];
  int nLocations;
} tracker_target_timed_location_array_t;

//TODO move to localizers/v0.c ? And move related definitions either to tracker.h or localizers/common.h ?
/**
 * @brief v0 - discrete localization using RSSI vector similarity
 */
static tracker_target_timed_location_array_t tracker_localize_v0(const tracker_mmvector_t* pInputs, 
                                                                 const tracker_physical_basepoint_t* pBasepoints,
                                                                 const int nInputs,
                                                                 const int nBasepoints) {
  int total_input_mvectors = 0;
  for (int i = 0; i < nInputs; i++) {
    total_input_mvectors += pInputs[i].nMvectors;
  }
  tracker_target_timed_location_array_t locations = {
    .nLocations = total_input_mvectors,
    .pLocations = {0}
  };
  assert(locations.nLocations <= TRACKER_TARGET_TIMED_LOCATION_ARRAY_MAXLEN);

  // Each input contains measurements from different i-cycles (inventory cycles)
  // For each input measurement, we first determine the most similar basepoint
  // Then we assign the (x, y, z) coordinates of that basepoint to the input location
  // We also assign the timestamp of the input measurement to the input timed location

  int lix = 0; // timed location index
  for (int i = 0; i < nInputs; i++) {
    for (int j = 0; j < pInputs[i].nMvectors; j++, lix++) {
      int max_similarity = -1;
      int max_similarity_basepoint_ix = -1;
      for (int k = 0; k < nBasepoints; k++) {
        int similarity = 0;
        for (int l = 0; l < pInputs[i].mvectors[j].componentCount; l++) {
          similarity += abs(pInputs[i].mvectors[j].rssi[l] - pBasepoints[k].mmvector.mvectors[j].rssi[l]);
        }
        if (similarity > max_similarity) {
          max_similarity = similarity;
          max_similarity_basepoint_ix = k;
        }
      }
      assert(max_similarity_basepoint_ix >= 0 && max_similarity_basepoint_ix < nBasepoints);
      locations.pLocations[lix].location = pBasepoints[max_similarity_basepoint_ix].point;
      assert(strncpy(locations.pLocations[lix].timestamp, pInputs[i].mvectors[j].timestamp, TRACKER_TIMESTAMP_LEN_MAX+1) == locations.pLocations[lix].timestamp);
      locations.pLocations[lix].pIob = (tracker_mmvector_t*)&pInputs[i];
    }
  }

  return locations;
}

//TODO add data validation to check if the cycles are in the correct order and contain the correct number of records?
static int __tracker_process_data_get_aerial_cycle_length(const int nInvm, const char** pAntnos) {
  assert(nInvm >= 0);
  assert(pAntnos != NULL);
  int cycleLength = 0;
  if (nInvm == 0) {
    return 0;
  }
  assert(pAntnos[0] != NULL);
  if (strcmp(pAntnos[0], "0") != 0) {
    LOG_E("__tracker_process_data_get_aerial_cycle_length: First record is not a sentry record, because antno is not 0");
    return 0;
  }
  for (int i = 1; i < nInvm; i++) {
    assert(pAntnos[i] != NULL);
    if (strcmp(pAntnos[i], "0") == 0) {
      cycleLength = i;
      break;
    }
  }
  return cycleLength;
}

static void tracker_invm_free(const int nInvm,
                              const char** pTimes, 
                              const char** pEpcs, 
                              const char** pAntnos, 
                              const char** pRxsss, 
                              const char** pRxrates, 
                              const char** pTxps,
                              const char** pRxlats, 
                              const char** pMtypes, 
                              const char** pRkts, 
                              const char** pRkps) {
  for (int i = 0; i < nInvm; i++) {
    free((void*)pTimes[i]);
    free((void*)pEpcs[i]);
    free((void*)pAntnos[i]);
    free((void*)pRxsss[i]);
    free((void*)pRxrates[i]);
    free((void*)pTxps[i]);
    free((void*)pRxlats[i]);
    free((void*)pMtypes[i]);
    free((void*)pRkts[i]);
    free((void*)pRkps[i]);
  }
  free((void*)pTimes);
  free((void*)pEpcs);
  free((void*)pAntnos);
  free((void*)pRxsss);
  free((void*)pRxrates);
  free((void*)pTxps);
  free((void*)pRxlats);
  free((void*)pMtypes);
  free((void*)pRkts);
  free((void*)pRkps);
}

#define TRACKER_BASEPOINTS_LIMIT 256

// TODO implement epc sanitization in lsapi to protect against malicious labs!
//                         Data format and cyclic structure:
//            A A B B C C D D E E A A B B C C D D E E A A B B C C D D E E 
// a-cycles  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | 
// i-cycles  |                   |                   |                   |
// m-chunk   |                                                           |
// a-cycle (aerial cycle) length is 2 in above example and corresponds to nComponents
// i-cycle (inventory cycle) length is 4 in above example
// m-chunk(measurements chunk) size is 30 in above example and corresponds to nInvm
//
// It's worth noting that nCycles = <number of i-cycles in m-chunk>
// In the above example, nInputs = 5 (A, B, C, D, E)
// number unique inventory items = nInvm / nCycles / nComponents (in above example, = 30 / 3 / 2 = 5)
// nInputs = nInvm / nCycles / nComponents - nBasepoints
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
                         const char** pRkps) {
  assert(pTracker != NULL);
  db_t* pDb = pTracker->pDb;
  assert(pDb != NULL);
  if (nInvm > 0) {
    assert(pTimes != NULL);
    assert(pEpcs != NULL);
    assert(pAntnos != NULL);
    assert(pRxsss != NULL);
    assert(pRxrates != NULL);
    assert(pTxps != NULL);
    assert(pRxlats != NULL);
    assert(pMtypes != NULL);
    assert(pRkts != NULL);
    assert(pRkps != NULL);

    LOG_V("tracker_process_data: nInvm = %d, nCycles = %d", nInvm, nCycles);
    assert(nCycles > 0);
    assert(nInvm % nCycles == 0);
    //const int nComponents = nInvm / nCycles; // wrong
    const int nComponents = __tracker_process_data_get_aerial_cycle_length(nInvm, pAntnos);
    assert(nComponents <= TRACKER_MVECTOR_NDIMS_MAX);
    db_lab_t lab;
    int rv = __tracker_get_lab(pDb, pEpcs[0], &lab);
    if (0 != rv) {
      LOG_E("tracker_process_data: __tracker_get_lab failed: %d", rv);
      return -1;
    }
    
    //Obtain basepoints (non-virtual) - DB_BASEPOINT_FILTER_TYPE_LID_NONVIRT
    int nBasepoints = 0;
    db_basepoint_t* pBasepoints = NULL;
    const char* _basepoints_page_offset_str = "0";
    const char* _basepoints_page_size_str = __tracker_itoa(TRACKER_BASEPOINTS_LIMIT + 1);
    const char* _lid_str = __tracker_itoa(lab.lab_id);
    rv = db_basepoints_read_page_filtered(pDb, 
                                          _basepoints_page_offset_str, 
                                          _basepoints_page_size_str, 
                                          &pBasepoints,
                                          &nBasepoints, 
                                          DB_BASEPOINT_FILTER_TYPE_LID_NONVIRT_EXT, 
                                          _lid_str);
    if (0 != rv) {
      LOG_E("tracker_process_data: db_basepoints_read_page_filtered failed: %d", rv);
      return -2;
    }
    if (nBasepoints > TRACKER_BASEPOINTS_LIMIT) {
      LOG_E("tracker_process_data: Too many basepoints for lab #%d. Current limit is %d", lab.lab_id, TRACKER_BASEPOINTS_LIMIT);
      //tracker_invm_free(nInvm, pTimes, pEpcs, pAntnos, pRxsss, pRxrates, pTxps, pRxlats, pMtypes, pRkts, pRkps);
      return -3;
    }
    if (nBasepoints == 0) {
      LOG_W("tracker_process_data: No basepoints for lab #%d. Abandoning localization", lab.lab_id);
      return 0;
    }
    for (int i = 0; i < nBasepoints; i++) {
      assert(pBasepoints[i].ext.epc != NULL);
    }

    const int localizationResults_Max = nInvm * TRACKER_TARGET_TIMED_LOCATION_ARRAY_MAXLEN;
    db_localization_result_t* pLocalizationResults = (db_localization_result_t*)malloc(localizationResults_Max * sizeof(db_localization_result_t));
    assert(pLocalizationResults != NULL);
    int localizationResults_counter = 0;

    //Process data
    tracker_physical_basepoint_t* pTrackerBasepoints = (tracker_physical_basepoint_t*)malloc(nBasepoints * sizeof(tracker_physical_basepoint_t));
    assert(pTrackerBasepoints != NULL);
    for (int i = 0; i < nBasepoints; i++) {
      pTrackerBasepoints[i].point.x = pBasepoints[i].x;
      pTrackerBasepoints[i].point.y = pBasepoints[i].y;
      pTrackerBasepoints[i].point.z = pBasepoints[i].z;
      pTrackerBasepoints[i].mmvector.nMvectors = nCycles;
      pTrackerBasepoints[i].mmvector.mvectors = (tracker_mvector_t*)malloc(nCycles * sizeof(tracker_mvector_t));
      assert(pTrackerBasepoints[i].mmvector.mvectors != NULL);
    }

    //int mvector_ix = -1;
    int* pMvectorIxs_Basepoints = (int*)malloc(nBasepoints * sizeof(int));
    assert(pMvectorIxs_Basepoints != NULL);
    for (int i = 0; i < nBasepoints; i++) {
      pMvectorIxs_Basepoints[i] = -1;
    }
    int aerial_cycle_counter_bp = -1;
    for (int i = 0; i < nInvm; i++) {
      //check if it's a basepoint
      assert(pEpcs[i] != NULL);
      for (int j = 0; j < nBasepoints; j++) {
        if (strcmp(pEpcs[i], pBasepoints[j].ext.epc) == 0) {
          if (strcmp(pAntnos[i], "0") == 0) {
            aerial_cycle_counter_bp++;
            if (aerial_cycle_counter_bp == nBasepoints) {
              aerial_cycle_counter_bp = 0;
            }
            pMvectorIxs_Basepoints[aerial_cycle_counter_bp]++;
            const int mvector_ix = pMvectorIxs_Basepoints[aerial_cycle_counter_bp];
            LOG_V("#sboji tracker_process_data:i = %d, j = %d, mvector_ix = %d, nCycles = %d, aerial_cycle_counter_bp = %d", i, j, mvector_ix, nCycles, aerial_cycle_counter_bp);
            assert(mvector_ix >= 0 && mvector_ix < nCycles);
            //LOG_V("tracker_process_data:j = %d, mvector_ix = %d", j, mvector_ix);
            pTrackerBasepoints[j].mmvector.mvectors[mvector_ix].componentCount = nComponents;
          }
          const int mvector_ix = pMvectorIxs_Basepoints[aerial_cycle_counter_bp];
          LOG_V("tracker_process_data: i = %d, j = %d, mvector_ix = %d, nCycles = %d, aerial_cycle_counter_bp = %d", i, j, mvector_ix, nCycles, aerial_cycle_counter_bp);
          assert(mvector_ix >= 0 && mvector_ix < nCycles);
          
          const int component_ix = atoi(pAntnos[i]);
          assert(component_ix >= 0 && component_ix <= TRACKER_MVECTOR_NDIMS_MAX);
          pTrackerBasepoints[j].mmvector.mvectors[mvector_ix].rssi[component_ix] = atoi(pRxsss[i]);
          assert(strlen(pTimes[i]) <= TRACKER_TIMESTAMP_LEN_MAX);
          assert(strncpy(pTrackerBasepoints[j].mmvector.mvectors[mvector_ix].timestamp, pTimes[i], TRACKER_TIMESTAMP_LEN_MAX+1) == pTrackerBasepoints[j].mmvector.mvectors[mvector_ix].timestamp);
          break;
        }
      }
    }

    assert(nBasepoints > 0);
    //assert(pTrackerBasepoints[0].mvector.componentCount > 0);
    // for (int i = 0; i < nBasepoints - 1; i++) {
    //   assert(pTrackerBasepoints[i].mvector.componentCount == pTrackerBasepoints[i + 1].mvector.componentCount);
    // }
    for (int i = 0; i < nBasepoints; i++) {
      assert(pTrackerBasepoints[i].mmvector.nMvectors == nCycles);
      assert(pTrackerBasepoints[i].mmvector.mvectors != NULL);
      for (int j = 0; j < nCycles; j++) {
        LOG_V("tracker_process_data: pTrackerBasepoints[i].mmvector.mvectors[j].componentCount = %d", pTrackerBasepoints[i].mmvector.mvectors[j].componentCount);
        LOG_V("tracker_process_data: nComponents = %d", nComponents);
        LOG_V("tracker_process_data: i = %d, j = %d", i, j);
        assert(pTrackerBasepoints[i].mmvector.mvectors[j].componentCount == nComponents);
      }
    }
    //index the tracked records (inputs) in a similar way as the basepoints
    assert(nCycles > 0); //TODO ensure malicious actors don't cause assertion failure
    assert(nComponents > 0); //TODO ensure malicious actors don't cause assertion failure
    const int nInputs = nInvm/nCycles/nComponents - nBasepoints;
    tracker_mmvector_t* pInputs = (tracker_mmvector_t*)malloc(nInputs * sizeof(tracker_mmvector_t));
    char** pInputEpcs = (char**)malloc(nInputs * sizeof(char*));
    assert(pInputs != NULL);
    assert(pInputEpcs != NULL);
    for (int i = 0; i < nInputs; i++) {
      pInputs[i].nMvectors = nCycles;
      pInputs[i].mvectors = (tracker_mvector_t*)malloc(nCycles * sizeof(tracker_mvector_t));
      assert(pInputs[i].mvectors != NULL);
    }
    //mvector_ix = -1;
    int* pMvectorIxs_Inputs = (int*)malloc(nInputs * sizeof(int));
    assert(pMvectorIxs_Inputs != NULL);
    for (int i = 0; i < nInputs; i++) {
      pMvectorIxs_Inputs[i] = -1;
    }
    int aerial_cycle_counter_trg = -1;
    int input_ix = -1;
    for (int i = 0; i < nInvm; i++) {
      //check if it's a basepoint
      assert(pEpcs[i] != NULL);
      int is_basepoint = 0;
      for (int j = 0; j < nBasepoints; j++) {
        if (strcmp(pEpcs[i], pBasepoints[j].ext.epc) == 0) {
          is_basepoint = 1;
          break;
        }
      }
      if (!is_basepoint) {
        input_ix++;
        //pInputEpcs[input_ix] = p_strdup(pEpcs[i]); //#osidvid
        if (strcmp(pAntnos[i], "0") == 0) {
          aerial_cycle_counter_trg++;
          if (aerial_cycle_counter_trg == nInputs) {
            aerial_cycle_counter_trg = 0;
          }
          pMvectorIxs_Inputs[aerial_cycle_counter_trg]++;
          //const int mvector_ix = pMvectorIxs_Inputs[input_ix];
          const int mvector_ix = pMvectorIxs_Inputs[aerial_cycle_counter_trg];
          LOG_V("tracker_process_data: #voidj i = %d, aerial_cycle_counter_trg = %d, mvector_ix = %d, nCycles = %d", i, aerial_cycle_counter_trg, mvector_ix, nCycles);
          assert(mvector_ix >= 0 && mvector_ix < nCycles);
          //pInputs[input_ix].mvectors[mvector_ix].componentCount = nComponents;
          pInputs[aerial_cycle_counter_trg].mvectors[mvector_ix].componentCount = nComponents;
        }
        const int mvector_ix = pMvectorIxs_Inputs[aerial_cycle_counter_trg]; // TODO input_ix and aerial_cycle_counter_trg are the same (double-check it), so we should probably refactor it for clarity
        assert(mvector_ix >= 0 && mvector_ix < nCycles);

        pInputEpcs[aerial_cycle_counter_trg] = p_strdup(pEpcs[i]); //@osidvid
        
        const int component_ix = atoi(pAntnos[i]);
        assert(component_ix >= 0 && component_ix <= TRACKER_MVECTOR_NDIMS_MAX);
        //pInputs[input_ix].mvectors[mvector_ix].rssi[component_ix] = atoi(pRxsss[i]);
        pInputs[aerial_cycle_counter_trg].mvectors[mvector_ix].rssi[component_ix] = atoi(pRxsss[i]);
        assert(strlen(pTimes[i]) <= TRACKER_TIMESTAMP_LEN_MAX);
        //assert(strncpy(pInputs[input_ix].mvectors[mvector_ix].timestamp, pTimes[i], TRACKER_TIMESTAMP_LEN_MAX+1) == pInputs[input_ix].mvectors[mvector_ix].timestamp);
        assert(strncpy(pInputs[aerial_cycle_counter_trg].mvectors[mvector_ix].timestamp, pTimes[i], TRACKER_TIMESTAMP_LEN_MAX+1) == pInputs[aerial_cycle_counter_trg].mvectors[mvector_ix].timestamp);
      }
    }

    // Call tracker localize function
    tracker_target_timed_location_array_t targetLocations = tracker_localize_v0(pInputs, pTrackerBasepoints, nInputs, nBasepoints);
    assert(targetLocations.nLocations >= 0);
    if (targetLocations.nLocations == 0) {
      LOG_W("tracker_process_data: No target locations for insertion");
      //TODO return earlier? (but any neccessary cleanup still needs to be done)
    }

    //Call db_localization_result_insert_bulk
    char** pTimesToInsert = (char**)malloc(targetLocations.nLocations * sizeof(char*));
    char** pEpcsToInsert = (char**)malloc(targetLocations.nLocations * sizeof(char*));
    char** pXsToInsert = (char**)malloc(targetLocations.nLocations * sizeof(char*));
    char** pYsToInsert = (char**)malloc(targetLocations.nLocations * sizeof(char*));
    char** pZsToInsert = (char**)malloc(targetLocations.nLocations * sizeof(char*));
    assert(pTimesToInsert != NULL);
    assert(pXsToInsert != NULL);
    assert(pYsToInsert != NULL);
    assert(pZsToInsert != NULL);
    for (int i = 0; i < targetLocations.nLocations; i++) {
      assert(strlen(targetLocations.pLocations[i].timestamp) <= TRACKER_TIMESTAMP_LEN_MAX);
      pTimesToInsert[i] = p_strdup(targetLocations.pLocations[i].timestamp);
      int epcIndex = (int)(targetLocations.pLocations[i].pIob - pInputs);
      assert(epcIndex >= 0 && epcIndex < nInputs);
      pEpcsToInsert[i] = pInputEpcs[epcIndex];
      pXsToInsert[i] = __tracker_dtoa((double)targetLocations.pLocations[i].location.x);
      pYsToInsert[i] = __tracker_dtoa((double)targetLocations.pLocations[i].location.y);
      pZsToInsert[i] = __tracker_dtoa((double)targetLocations.pLocations[i].location.z);
    }

    // ///<tmp_test>
    // free (pInputEpcs);
    // pInputEpcs = (char**)malloc(2 * sizeof(char*));
    // assert(pInputEpcs != NULL);
    // pInputEpcs[0] = "aaaaaaaaaaaaaaaaaaaaaaaa";
    // pInputEpcs[1] = "bbbbbbbbbbbbbbbbbbbbbbbb";
    // ///</tmp_test>
    rv = db_localization_result_insert_bulk(pDb, targetLocations.nLocations, (const char**)pTimesToInsert, (const char**)pEpcsToInsert, (const char**)pXsToInsert, (const char**)pYsToInsert, (const char**)pZsToInsert);
    for (int i = 0; i < nBasepoints; i++) {
      db_basepoint_free(&pBasepoints[i]);
    }
    free(pBasepoints);
    for (int i = 0; i < nBasepoints; i++) {
      free(pTrackerBasepoints[i].mmvector.mvectors);
    }
    free(pTrackerBasepoints);
    for (int i = 0; i < nInputs; i++) {
      free(pInputs[i].mvectors);
    }
    free(pInputs);
    free(pInputEpcs);
    for (int i = 0; i < targetLocations.nLocations; i++) {
      free(pTimesToInsert[i]);
      free(pXsToInsert[i]);
      free(pYsToInsert[i]);
      free(pZsToInsert[i]);
    }
    free(pTimesToInsert);
    free(pEpcsToInsert);
    free(pXsToInsert);
    free(pYsToInsert);
    free(pZsToInsert);

    //tracker_invm_free(nInvm, pTimes, pEpcs, pAntnos, pRxsss, pRxrates, pTxps, pRxlats, pMtypes, pRkts, pRkps);
    free(pLocalizationResults);
    free(pMvectorIxs_Basepoints);
    free(pMvectorIxs_Inputs);
    if (0 != rv) {
      LOG_E("tracker_process_data: db_localization_result_insert_bulk failed: %d", rv);
      return -8;
    }
    return 0;
  } else {
    LOG_W("tracker_process_data: No inventory data to process");
    return 0;
  }
}

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
                                  const int* pIsSentry) {
  assert(pTracker != NULL);
  db_t* pDb = pTracker->pDb;
  assert(pDb != NULL);
  if (nInvm > 0) {
    assert(pTimes != NULL);
    assert(pEpcs != NULL);
    assert(pAntnos != NULL);
    assert(pRxsss != NULL);
    assert(pRxrates != NULL);
    assert(pTxps != NULL);
    assert(pRxlats != NULL);
    assert(pMtypes != NULL);
    assert(pRkts != NULL);
    assert(pRkps != NULL);
    assert(pIsSentry != NULL);
    db_lab_t lab;
    int rv = __tracker_get_lab(pDb, pEpcs[0], &lab);
    if (0 != rv) {
      LOG_E("tracker_process_data: __tracker_get_lab failed: %d", rv);
      return -1;
    }
    
    tracker_lab_buffer_t* pLabBuffer = tracker_lab_buffer_find(pTracker, lab.lab_id);
    if (pLabBuffer == NULL) {
      assert(TRUE == p_mutex_lock(pTracker->pMutex));
      pTracker->nLabBuffers++;
      pTracker->pLabBuffers = (tracker_lab_buffer_t*)realloc(pTracker->pLabBuffers, pTracker->nLabBuffers * sizeof(tracker_lab_buffer_t));
      if (pTracker->pLabBuffers == NULL) {
        LOG_E("tracker_process_data_buffered: realloc failed");
        db_lab_free(&lab);
        assert(TRUE == p_mutex_unlock(pTracker->pMutex));
        return -2;
      }
      pLabBuffer = &pTracker->pLabBuffers[pTracker->nLabBuffers - 1];
      *pLabBuffer = tracker_lab_buffer_new(lab.lab_id);
      assert(TRUE == p_mutex_unlock(pTracker->pMutex));
    }
    assert(pLabBuffer != NULL);
    assert(pLabBuffer->pMutex != NULL);

    assert(TRUE == p_mutex_lock(pLabBuffer->pMutex));

    // Process entries
    int* sentry_positions = (int*)malloc(nInvm * sizeof(int));
    assert(sentry_positions != NULL);
    int nSentries = 0;
    for (int i = 0; i < nInvm; i++) {
      if (pIsSentry[i]) {
        sentry_positions[nSentries++] = i;
      }
    }
    if (tracker_lab_buffer_is_empty(pLabBuffer)) { //should occur only once
      if (nSentries == 0) {
        LOG_W("tracker_process_data_buffered: No sentries found in telemetry data even though the lab buffer is empty. Discarding data");
        free(sentry_positions);
        db_lab_free(&lab);
        assert(TRUE == p_mutex_unlock(pLabBuffer->pMutex));
        return 0;
      } else if (nSentries == 1) {
        LOG_I("tracker_process_data_buffered: Only one sentry found in telemetry data, thus the data will be buffered until the next sentry is found");
        //concatenate the data to the lab buffer
        tracker_lab_buffer_concat(pLabBuffer, nInvm, pTimes, pEpcs, pAntnos, pRxsss, pRxrates, pTxps, pRxlats, pMtypes, pRkts, pRkps);
        free(sentry_positions);
        db_lab_free(&lab);
        assert(TRUE == p_mutex_unlock(pLabBuffer->pMutex));
        return 0;
      } else {
        LOG_I("tracker_process_data_buffered: Multiple sentries found in telemetry data, thus the data will be processed immediately");
        //immediately process data up to the last sentry (exclusive)
        int nInvmToProcess = sentry_positions[nSentries - 1];
        rv = tracker_process_data(pTracker, nInvmToProcess, nSentries - 1, pTimes, pEpcs, pAntnos, pRxsss, pRxrates, pTxps, pRxlats, pMtypes, pRkts, pRkps);
        if (rv != 0) {
          LOG_E("tracker_process_data_buffered: tracker_process_data failed: %d", rv);
          free(sentry_positions);
          db_lab_free(&lab);
          assert(TRUE == p_mutex_unlock(pLabBuffer->pMutex));
          return -3;
        }
        //concatenate the remaining data to the lab buffer
        tracker_lab_buffer_concat(pLabBuffer, nInvm - nInvmToProcess, pTimes + nInvmToProcess, pEpcs + nInvmToProcess, pAntnos + nInvmToProcess, pRxsss + nInvmToProcess, pRxrates + nInvmToProcess, pTxps + nInvmToProcess, pRxlats + nInvmToProcess, pMtypes + nInvmToProcess, pRkts + nInvmToProcess, pRkps + nInvmToProcess);

        free(sentry_positions);
        db_lab_free(&lab);
        assert(TRUE == p_mutex_unlock(pLabBuffer->pMutex));
        return 0;
      }
    } else {
      if (nSentries == 0) {
        LOG_W("tracker_process_data_buffered: No sentries found in telemetry data. Buffering data");
        //concatenate the data to the lab buffer
        tracker_lab_buffer_concat(pLabBuffer, nInvm, pTimes, pEpcs, pAntnos, pRxsss, pRxrates, pTxps, pRxlats, pMtypes, pRkts, pRkps);
        free(sentry_positions);
        db_lab_free(&lab);
        assert(TRUE == p_mutex_unlock(pLabBuffer->pMutex));
        return 0;
      } else {
        //add to buffer for cyclic completion and process data up to the last sentry (exclusive)
        int nInvmToProcess = sentry_positions[nSentries - 1];
        tracker_lab_buffer_concat(pLabBuffer, nInvmToProcess, pTimes, pEpcs, pAntnos, pRxsss, pRxrates, pTxps, pRxlats, pMtypes, pRkts, pRkps);
        rv = tracker_process_data(pTracker, pLabBuffer->buf_len, nSentries, (const char**)pLabBuffer->pTimes, (const char**)pLabBuffer->pEpcs, (const char**)pLabBuffer->pAntnos, (const char**)pLabBuffer->pRxsss, (const char**)pLabBuffer->pRxrates, (const char**)pLabBuffer->pTxps, (const char**)pLabBuffer->pRxlats, (const char**)pLabBuffer->pMtypes, (const char**)pLabBuffer->pRkts, (const char**)pLabBuffer->pRkps);
        if (rv != 0) {
          LOG_E("tracker_process_data_buffered: tracker_process_data failed: %d", rv);
          free(sentry_positions);
          db_lab_free(&lab);
          assert(TRUE == p_mutex_unlock(pLabBuffer->pMutex));
          return -4;
        }
        //tracker_lab_buffer_unbuffer(pLabBuffer, pLabBuffer->buf_len);
        tracker_lab_buffer_clear(pLabBuffer);
        //concatenate the remaining data to the lab buffer
        tracker_lab_buffer_concat(pLabBuffer, nInvm - nInvmToProcess, pTimes + nInvmToProcess, pEpcs + nInvmToProcess, pAntnos + nInvmToProcess, pRxsss + nInvmToProcess, pRxrates + nInvmToProcess, pTxps + nInvmToProcess, pRxlats + nInvmToProcess, pMtypes + nInvmToProcess, pRkts + nInvmToProcess, pRkps + nInvmToProcess);

        free(sentry_positions);
        db_lab_free(&lab);
        assert(TRUE == p_mutex_unlock(pLabBuffer->pMutex));
        return 0;
      }
    }
  } else {
    LOG_W("tracker_process_data_buffered: No inventory data to process");
    return 0;
  }
}

tracker_t* tracker_new(db_t* pDb) {
  tracker_t* pTracker = (tracker_t*)malloc(sizeof(tracker_t));
  assert(pTracker != NULL);
  pTracker->pDb = pDb;
  pTracker->pLabBuffers = NULL;
  pTracker->nLabBuffers = 0;

  pTracker->pMutex = p_mutex_new();
  assert(pTracker->pMutex != NULL);
  return pTracker;
}

void tracker_free(tracker_t* pTracker) {
  assert(pTracker != NULL);
  if (pTracker->pLabBuffers != NULL) {
    for (int i = 0; i < pTracker->nLabBuffers; i++) {
      //TODO replace with tracker_lab_buffer_free?
      if (pTracker->pLabBuffers[i].pTimes != NULL) {
        for (int j = 0; j < pTracker->pLabBuffers[i].buf_len; j++) {
          free(pTracker->pLabBuffers[i].pTimes[j]);
          free(pTracker->pLabBuffers[i].pEpcs[j]);
          free(pTracker->pLabBuffers[i].pAntnos[j]);
          free(pTracker->pLabBuffers[i].pRxsss[j]);
          free(pTracker->pLabBuffers[i].pRxrates[j]);
          free(pTracker->pLabBuffers[i].pTxps[j]);
          free(pTracker->pLabBuffers[i].pRxlats[j]);
          free(pTracker->pLabBuffers[i].pMtypes[j]);
          free(pTracker->pLabBuffers[i].pRkts[j]);
          free(pTracker->pLabBuffers[i].pRkps[j]);
        }
        free(pTracker->pLabBuffers[i].pTimes);
        free(pTracker->pLabBuffers[i].pEpcs);
        free(pTracker->pLabBuffers[i].pAntnos);
        free(pTracker->pLabBuffers[i].pRxsss);
        free(pTracker->pLabBuffers[i].pRxrates);
        free(pTracker->pLabBuffers[i].pTxps);
        free(pTracker->pLabBuffers[i].pRxlats);
        free(pTracker->pLabBuffers[i].pMtypes);
        free(pTracker->pLabBuffers[i].pRkts);
        free(pTracker->pLabBuffers[i].pRkps);
      }
    }
    free(pTracker->pLabBuffers);
  }
  assert(pTracker->pMutex != NULL);
  p_mutex_free(pTracker->pMutex);
  free(pTracker);
}
