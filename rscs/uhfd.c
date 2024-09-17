#include "uhfd.h"

#include <time.h>
#include <stdlib.h>
#include <plibsys.h>
#include <pmacros.h>
#include <assert.h>
#include "uhfman.h"
#include "log.h"

//MTRand __uhfd_mtrand;

// void uhfd_combined_lock(uhfd_t* pUHFD, uhfd_combined_lock_flags_t flags) {
//     if (flags & UHFD_LOCK_FLAG_STATE_MUTEX) {
//         assert(TRUE == p_mutex_lock(pUHFD->pStateMutex));
//     }
//     if (flags & UHFD_LOCK_FLAG_DEVCSIZE_MUTEX) {
//         assert(TRUE == p_mutex_lock(pUHFD->pDevCSizeMutex));
//     }
//     if (flags & UHFD_LOCK_FLAG_DEV_RWLOCK_WRITE) {
//         assert(TRUE == p_rwlock_writer_lock(pUHFD->pDevRWLock));
//     } else if (flags & UHFD_LOCK_FLAG_DEV_RWLOCK_READ) {
//         assert(TRUE == p_rwlock_reader_lock(pUHFD->pDevRWLock));
//     }
// }

// void uhfd_combined_unlock(uhfd_t* pUHFD, uhfd_combined_lock_flags_t flags) {
//     if (flags & UHFD_LOCK_FLAG_STATE_MUTEX) {
//         assert(TRUE == p_mutex_unlock(pUHFD->pStateMutex));
//     }
//     if (flags & UHFD_LOCK_FLAG_DEVCSIZE_MUTEX) {
//         assert(TRUE == p_mutex_unlock(pUHFD->pDevCSizeMutex));
//     }
//     if (flags & UHFD_LOCK_FLAG_DEV_RWLOCK_WRITE) {
//         assert(TRUE == p_rwlock_writer_unlock(pUHFD->pDevRWLock));
//     } else if (flags & UHFD_LOCK_FLAG_DEV_RWLOCK_READ) {
//         assert(TRUE == p_rwlock_reader_unlock(pUHFD->pDevRWLock));
//     }
// }

/*  No need to do atomics or locks here as the function is supposed to be called only once and no other function is supposed to be running at the same time
    ~~No need to lock here as the function is supposed to be called only once~~
*/
int uhfd_init(uhfd_t* pUHFD_out) {

    //__uhfd_mtrand = seedRand(time(NULL));
    
    // pUHFD_out->pInitDeinitMutex = p_mutex_new();
    // assert(pUHFD_out->pInitDeinitMutex != NULL);
    // p_mutex_lock(pUHFD_out->pInitDeinitMutex);

    pUHFD_out->state_flags &= ~(uint8_t)UHFD_STATE_FLAG_INITIALIZED; // defensive
    // pUHFD_out->pStateMutex = p_mutex_new();
    // assert(pUHFD_out->pStateMutex != NULL);
    // pUHFD_out->pDevCSizeMutex = p_mutex_new();
    // assert(pUHFD_out->pDevCSizeMutex != NULL);
    // pUHFD_out->pDevRWLock = p_rwlock_new();
    // assert(pUHFD_out->pDevRWLock != NULL);
    pUHFD_out->pDaRWLock = p_rwlock_new();
    assert(pUHFD_out->pDaRWLock != NULL);

    //p_mutex_lock(pUHFD_out->pStateMutex); // no need to lock yet

    pUHFD_out->mtrand = seedRand(time(NULL));

    // pUHFD_out->pDevCSizeMutex = p_mutex_new();
    // assert(pUHFD_out->pDevCSizeMutex != NULL);

    // pUHFD_out->num_devs_flex_max = UHFD_NUM_DEVS_INITIAL;
    // pUHFD_out->pDevs = (uhfd_dev_t*) malloc(UHFD_NUM_DEVS_INITIAL * sizeof(uhfd_dev_t));
    // assert(pUHFD_out->pDevs != NULL);
    // pUHFD_out->num_devs = 0;
    pUHFD_out->da.num_devs_flex_max = UHFD_NUM_DEVS_INITIAL;
    pUHFD_out->da.pDevs = (uhfd_dev_t*) malloc(UHFD_NUM_DEVS_INITIAL * sizeof(uhfd_dev_t));
    assert(pUHFD_out->da.pDevs != NULL);
    pUHFD_out->da.num_devs = 0;
    
    // pUHFD_out->pCreateDevMutex =  p_mutex_new();
    // assert(pUHFD_out->pCreateDevMutex != NULL); 
    // pUHFD_out->pDevRWLock = p_rwlock_new();
    // assert(pUHFD_out->pDevRWLock != NULL);
    
    pUHFD_out->uhfmanCtx = (uhfman_ctx_t){0};
    uhfman_err_t err = uhfman_device_take(&pUHFD_out->uhfmanCtx);
    if (err != UHFMAN_TAKE_ERR_SUCCESS) {
        LOG_E("uhfd_init: uhfman_device_take failed with error %d", err);
        return -1;
    }
    pUHFD_out->state_flags |= (uint8_t)UHFD_STATE_FLAG_INITIALIZED;
    //p_mutex_unlock(pUHFD_out->pInitDeinitMutex);
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
    
    uint8_t state_flags = (unsigned long)p_atomic_pointer_get(&pUHFD->state_flags);
    if (state_flags & (uint8_t)UHFD_STATE_FLAG_INITIALIZED) {
        // assert(TRUE == p_mutex_lock(pUHFD->pCreateDevMutex));
        // assert(TRUE == p_mutex_lock(pUHFD->pDevCSizeMutex));

        p_rwlock_writer_lock(pUHFD->pDaRWLock);
        uhfd_dev_array_t* pDa = &pUHFD->da;
        if (pDa->num_devs >= pDa->num_devs_flex_max) {
            pDa->num_devs_flex_max *= UHFD_NUM_DEVS_SCALING_FACTOR;
            // p_rwlock_writer_lock(pUHFD->pDevRWLock); // Need to write lock the entire array for realloc
            pDa->pDevs = (uhfd_dev_t*) realloc(pDa->pDevs, pDa->num_devs_flex_max * sizeof(uhfd_dev_t));
            assert(pDa->pDevs != NULL);
            // p_rwlock_writer_unlock(pUHFD->pDevRWLock);
        }
        *pDevNum_out = pDa->num_devs;
        uhfd_dev_t* pDev = &(pDa->pDevs[pDa->num_devs]);
        *pDev = (uhfd_dev_t) {
            .epc = {0},
            .access_passwd = {0},
            .kill_passwd = {0},
            .flags = 0x00,
            .measurement = (uhfd_dev_m_t) {
                .rssi = 0,
                .read_rate = 0
            },
            .devno = pDa->num_devs
        };
        pDa->num_devs++;
        p_rwlock_writer_unlock(pUHFD->pDaRWLock);
        //uhfd_dev_array_t da = (uhfd_dev_array_t)p_atomic_pointer_get(&pUHFD->da);
        // assert(TRUE == p_mutex_unlock(pUHFD->pCreateDevMutex));
        // assert(TRUE == p_mutex_unlock(pUHFD->pDevCSizeMutex));
        return 0;
    } else {
        return -1;
    }
}

// // @dev consider refactoring to uhfd_dev_get_synchronized - would require changing other function names as well
// int uhfd_get_dev_synchronized(uhfd_t* pUHFD, unsigned long devno, uhfd_dev_t* pDev_out) {
//     //assert(TRUE == p_rwlock_reader_lock(pUHFD->pDevRWLock));
//     assert(TRUE == p_rwlock_reader_lock(pUHFD->pDaRWLock));
//     uhfd_dev_array_t* pDa = &pUHFD->da;
//     if (devno < pDa->num_devs) {
//         *pDev_out = pDa->pDevs[devno];
//         //assert(TRUE == p_rwlock_reader_unlock(pUHFD->pDevRWLock));
//         assert(TRUE == p_rwlock_reader_unlock(pUHFD->pDaRWLock));
//         return 0;
//     } else {
//         //assert(TRUE == p_rwlock_reader_unlock(pUHFD->pDevRWLock));
//         assert(TRUE == p_rwlock_reader_unlock(pUHFD->pDaRWLock));
//         return -1;
//     }
// }

int uhfd_get_dev(uhfd_t* pUHFD, unsigned long devno, uhfd_dev_t* pDev_out) {
    assert(TRUE == p_rwlock_reader_lock(pUHFD->pDaRWLock));
    uhfd_dev_array_t* pDa = &pUHFD->da;
    if (devno < pDa->num_devs) {
        *pDev_out = pDa->pDevs[devno];
        assert(TRUE == p_rwlock_reader_unlock(pUHFD->pDaRWLock));
        return 0;
    } else {
        assert(TRUE == p_rwlock_reader_unlock(pUHFD->pDaRWLock));
        return -1;
    }
}

int uhfd_set_dev(uhfd_t* pUHFD, unsigned long devno, uhfd_dev_t dev) {
    assert(TRUE == p_rwlock_writer_lock(pUHFD->pDaRWLock));
    uhfd_dev_array_t* pDa = &pUHFD->da;
    if (devno < pDa->num_devs) {
        pDa->pDevs[devno] = dev;
        assert(TRUE == p_rwlock_writer_unlock(pUHFD->pDaRWLock));
        return 0;
    } else {
        assert(TRUE == p_rwlock_writer_unlock(pUHFD->pDaRWLock));
        return -1;
    }
}

// int uhfd_get_dev_flags_synchronized(uhfd_t* pUHFD, unsigned long devno, uint8_t* pFlags_out) {
//     assert(TRUE == p_rwlock_reader_lock(pUHFD->pDevRWLock));
//     if (devno < pUHFD->num_devs) {
//         *pFlags_out = pUHFD->pDevs[devno].flags;
//         assert(TRUE == p_rwlock_reader_unlock(pUHFD->pDevRWLock));
//         return 0;
//     } else {
//         assert(TRUE == p_rwlock_reader_unlock(pUHFD->pDevRWLock));
//         return -1;
//     }
// }

int uhfd_get_num_devs(uhfd_t* pUHFD, unsigned long* pNumDevs_out) {
    assert(TRUE == p_rwlock_reader_lock(pUHFD->pDaRWLock));
    *pNumDevs_out = pUHFD->da.num_devs;
    assert(TRUE == p_rwlock_reader_unlock(pUHFD->pDaRWLock));
    return 0;
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

/*  The function is supposed to be called only once and will not care about pending file operations if they even exist
    //~~(I'm not sure file operations' handlers can still be running while this function is called)~~
    //~~That's why we're locking defensively, unless you can ensure me that it's safe without that.~~
*/
int uhfd_deinit(uhfd_t* pUHFD) {
    //assert(TRUE == p_mutex_lock(pUHFD->pInitDeinitMutex));
    // uhfd_combined_lock_flags_t flags = UHFD_LOCK_FLAG_DEV_RWLOCK_WRITE | UHFD_LOCK_FLAG_DEVCSIZE_MUTEX | UHFD_LOCK_FLAG_STATE_MUTEX;
    // uhfd_combined_lock(pUHFD, flags); // Defensively wait for any file operations to finish. Is this good?
    // if (NULL != pUHFD->pDevs) {
    //     free(pUHFD->pDevs);
    // }
    uint8_t flags = (unsigned long)p_atomic_pointer_get(&pUHFD->state_flags);
    if (flags & (uint8_t)UHFD_STATE_FLAG_INITIALIZED) {
        LOG_E("uhfd_deinit: not initialized");
        return -1;
    }
    p_atomic_pointer_set(&pUHFD->state_flags, (void*)(unsigned long)(flags & ~(uint8_t)UHFD_STATE_FLAG_INITIALIZED));

    if (NULL != pUHFD->da.pDevs) {
        free(pUHFD->da.pDevs);
    }
    // pUHFD->num_devs = 0;
    // pUHFD->num_devs_flex_max = 0;
    pUHFD->da.num_devs = 0;
    pUHFD->da.num_devs_flex_max = 0;
    pUHFD->state_flags &= ~(uint8_t)UHFD_STATE_FLAG_INITIALIZED;
    //p_mutex_free(pUHFD->pCreateDevMutex);
    //p_rwlock_free(pUHFD->pDevRWLock);
    p_rwlock_free(pUHFD->pDaRWLock);

    // uhfd_combined_unlock(pUHFD, flags);

    uhfman_device_release(&pUHFD->uhfmanCtx);
    if (errno != 0) {
        LOG_E("uhfd_deinit: uhfman_device_release failed, errno=%d", errno);
        return -1;
    }
    //assert(TRUE == p_mutex_unlock(pUHFD->pInitDeinitMutex));
    //p_mutex_free(pUHFD->pInitDeinitMutex);
    return 0;
}