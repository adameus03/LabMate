#include "uhfd.h"

#include <time.h>
#include <stdlib.h>
#include <plibsys.h>
#include <pmacros.h>
#include <assert.h>
#include "uhfman.h"
#include "log.h"

//MTRand __uhfd_mtrand;

int uhfd_init(uhfd_t* pUHFD_out) {
    //__uhfd_mtrand = seedRand(time(NULL));
    pUHFD_out->pInitDeinitMutex = p_mutex_new();
    assert(pUHFD_out->pInitDeinitMutex != NULL);
    p_mutex_lock(pUHFD_out->pInitDeinitMutex);
    pUHFD_out->state_flags &= ~(uint8_t)UHFD_STATE_FLAG_INITIALIZED; // defensive
    pUHFD_out->mtrand = seedRand(time(NULL));
    pUHFD_out->num_devs_flex_max = UHFD_NUM_DEVS_INITIAL;
    pUHFD_out->pDevs = (uhfd_dev_t*) malloc(UHFD_NUM_DEVS_INITIAL * sizeof(uhfd_dev_t));
    assert(pUHFD_out->pDevs != NULL);
    pUHFD_out->num_devs = 0;
    pUHFD_out->pCreateDevMutex =  p_mutex_new();
    assert(pUHFD_out->pCreateDevMutex != NULL); 
    pUHFD_out->pDevRWLock = p_rwlock_new();
    assert(pUHFD_out->pDevRWLock != NULL);
    pUHFD_out->uhfmanCtx = (uhfman_ctx_t){0};
    uhfman_err_t err = uhfman_device_take(&pUHFD_out->uhfmanCtx);
    if (err != UHFMAN_TAKE_ERR_SUCCESS) {
        LOG_E("uhfd_init: uhfman_device_take failed with error %d", err);
        return -1;
    }
    pUHFD_out->state_flags |= (uint8_t)UHFD_STATE_FLAG_INITIALIZED;
    p_mutex_unlock(pUHFD_out->pInitDeinitMutex);
    return 0;
}

//int uhfd_create_dev(uhfd_t* pUHFD, uhfd_dev_t* pDev_out) {
int uhfd_create_dev(uhfd_t* pUHFD, unsigned long* pDevNum_out) {
    // // generate random 12B of EPC
    // uint8_t __unready = 1U;
    // do {
    //     for (uint8_t i=0U; i<12U; i++) {
    //         pDev_out->epc[i] = getRandLong(pUHFD->mtrand) & 0xFFU;
    //     }
    //     // ensure it's not all zeroes
    //     for (uint8_t i=0U; i<12U; i++) {
    //         if (pDev_out->epc[i] != 0U) {
    //             __unready = 0;
    //             // check if the epc is already used by any /custom/uhfX/epc
                
                
    //             break;
    //         }
    //     }
    // } while (__unready);
    
    if (pUHFD->state_flags & (uint8_t)UHFD_STATE_FLAG_INITIALIZED) {
        assert(TRUE == p_mutex_lock(pUHFD->pCreateDevMutex));
        if (pUHFD->num_devs >= pUHFD->num_devs_flex_max) {
            pUHFD->num_devs_flex_max *= UHFD_NUM_DEVS_SCALING_FACTOR;
            pUHFD->pDevs = (uhfd_dev_t*) realloc(pUHFD->pDevs, pUHFD->num_devs_flex_max * sizeof(uhfd_dev_t));
            assert(pUHFD->pDevs != NULL);
        }
        *pDevNum_out = pUHFD->num_devs;
        uhfd_dev_t* pDev = &(pUHFD->pDevs[pUHFD->num_devs]);
        *pDev = (uhfd_dev_t) {
            .epc = {0},
            .access_passwd = {0},
            .kill_passwd = {0},
            .flags = 0x00,
            .measurement = (uhfd_dev_m_t) {
                .rssi = 0,
                .read_rate = 0
            },
            .devno = pUHFD->num_devs
        };
        pUHFD->num_devs++;
        assert(TRUE == p_mutex_unlock(pUHFD->pCreateDevMutex));
        return 0;
    } else {
        return -1;
    }
}

// @dev consider refactoring to uhfd_dev_get_atomic - would require changing other function names as well
int uhfd_atomic_get_dev(uhfd_t* pUHFD, unsigned long devno, uhfd_dev_t* pDev_out) {
    assert(TRUE == p_rwlock_reader_lock(pUHFD->pDevRWLock));
    if (devno < pUHFD->num_devs) {
        *pDev_out = pUHFD->pDevs[devno];
        assert(TRUE == p_rwlock_reader_unlock(pUHFD->pDevRWLock));
        return 0;
    } else {
        assert(TRUE == p_rwlock_reader_unlock(pUHFD->pDevRWLock));
        return -1;
    }
}

int uhfd_atomic_get_dev_flags(uhfd_t* pUHFD, unsigned long devno, uint8_t* pFlags_out) {
    assert(TRUE == p_rwlock_reader_lock(pUHFD->pDevRWLock));
    if (devno < pUHFD->num_devs) {
        *pFlags_out = pUHFD->pDevs[devno].flags;
        assert(TRUE == p_rwlock_reader_unlock(pUHFD->pDevRWLock));
        return 0;
    } else {
        assert(TRUE == p_rwlock_reader_unlock(pUHFD->pDevRWLock));
        return -1;
    }
}

//should we even bother to kill the tag? (it may not be neccessary!)
int uhfd_delete_dev(uhfd_t* pUHFD, uhfd_dev_t* pDev, uint8_t flags) {
    // Set the 'deleted' flag for simplest solution
    LOG_E("uhfd_delete_dev: not implemented");
    return -1;
}

int uhfd_embody_dev(uhfd_t* pUHFD, uhfd_dev_t* pDev) {
    LOG_E("uhfd_embody_dev: not implemented");
    return -1;
}

int uhfd_measure_dev(uhfd_t* pUHFD, uhfd_dev_t* pDev, uhfd_dev_m_t* pMeasurement) {
    LOG_E("uhfd_measure_dev: not implemented");
    return -1;
}

int uhfd_deinit(uhfd_t* pUHFD) {
    assert(TRUE == p_mutex_lock(pUHFD->pInitDeinitMutex));
    if (NULL != pUHFD->pDevs) {
        free(pUHFD->pDevs);
    }
    pUHFD->num_devs = 0;
    pUHFD->num_devs_flex_max = 0;
    pUHFD->state_flags &= ~(uint8_t)UHFD_STATE_FLAG_INITIALIZED;
    p_mutex_free(pUHFD->pCreateDevMutex);
    p_rwlock_free(pUHFD->pDevRWLock);
    uhfman_device_release(&pUHFD->uhfmanCtx);
    if (errno != 0) {
        LOG_E("uhfd_deinit: uhfman_device_release failed, errno=%d", errno);
        return -1;
    }
    assert(TRUE == p_mutex_unlock(pUHFD->pInitDeinitMutex));
    p_mutex_free(pUHFD->pInitDeinitMutex);
    return 0;
}