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
    pUHFD_out->pUhfmanCtxMutex = p_mutex_new();
    assert(pUHFD_out->pUhfmanCtxMutex != NULL);

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
            .flags1 = 0x00,
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
int uhfd_delete_dev(uhfd_t* pUHFD, /*uhfd_dev_t* pDev*/unsigned long devno, uint8_t flags) {
    // Set the 'deleted' flag for simplest solution
    LOG_E("uhfd_delete_dev: not implemented");
    return -1;
}

//Designed to handle locking for tag devs and partial write (when passwords write succeeds, but EPC write fails)
int uhfd_embody_dev(uhfd_t* pUHFD, /*uhfd_dev_t* pDev*/unsigned long devno) {
    LOG_E("uhfd_embody_dev: not implemented");
    return -1;

    /* dev attention
    *  WE NEED TO FIND OUT WHETHER WE CAN WRITE THE TAG USING GENERIC OR SPECIFIC SELECT BEFOREHAND
        - if generic - less code, and possibly quicker? We need to set the select and query parameters such that we communicate with any tag, while the non-zero password can protect the tag from being written unintentionally
        - if specific - more code, and possibly more reliable? We need to poll for the first detected tag and then use it's EPC
    */
    // Old answer: we use the first option - generic select and query so that the tag can be written without any prior knowledge of it
    // New answer: we use the second option, because we want to be safe as the tag is written multiple times in the process (EPC and RESERVED membanks) and we want to avoid writing to a wrong tag
    // Even newer answer: If the application makes sure the access passwords are unique, that should not be a problem. We will thus use the first approach

    // [skip] Poll for first detected tag
    // [skip] Read tag EPC
    // [skip] Set select and query parameters such that we communicate only with that tag (see old_main.c)
    // [skip]Check if the EPC is already used by any tag and handle that (error? or warning? or not?)
    
    // Set generic query parameters, select mode, select params, tx power (smaller range to avoid unwanted writes)
    uint8_t select_target = UHFMAN_SELECT_TARGET_SL;
    uint8_t select_action = uhfman_select_action(UHFMAN_SEL_SL_ASSERT, UHFMAN_SEL_SL_DEASSERT);
    assert(select_action != UHFMAN_SELECT_ACTION_UNKNOWN);
    uint8_t select_memBank = UHFMAN_SELECT_MEMBANK_EPC;
    uint8_t select_ptr = 0x20;
    uint8_t select_maskLen = 0x00;
    uint8_t select_truncate = UHFMAN_SELECT_TRUNCATION_DISABLED;
    uint8_t select_mask[0] = {};
    uhfman_query_sel_t query_sel = UHFMAN_QUERY_SEL_SL;
    uhfman_query_session_t query_session = UHFMAN_QUERY_SESSION_S0;
    uhfman_query_target_t query_target = UHFMAN_QUERY_TARGET_A;
    uint8_t query_q = 0x00;
    uhfman_select_mode_t select_mode = UHFMAN_SELECT_MODE_ALWAYS;
    float txPower = 15.0f; // for now 15dBm, but we can make setting it 0 automatically use the minimum power for specific interrogator hardware (in the future)
    
    assert(TRUE == p_mutex_lock(pUHFD->pUhfmanCtxMutex));
    uhfman_err_t err = uhfman_set_select_param(&pUHFD->uhfmanCtx, select_target, select_action, select_memBank, select_ptr, select_maskLen, select_truncate, select_mask);
    if (err != UHFMAN_SET_SELECT_PARAM_ERR_SUCCESS) {
        LOG_E("uhfd_embody_dev: uhfman_set_select_param failed with error %d", err);
        return -1;
    }
    err = uhfman_set_query_params(&pUHFD->uhfmanCtx, query_sel, query_session, query_target, query_q);
    if (err != UHFMAN_SET_QUERY_PARAMS_ERR_SUCCESS) {
        LOG_E("uhfd_embody_dev: uhfman_set_query_params failed with error %d", err);
        return -1;
    }
    err = uhfman_set_select_mode(&pUHFD->uhfmanCtx, select_mode);
    if (err != UHFMAN_SET_SELECT_MODE_ERR_SUCCESS) {
        LOG_E("uhfd_embody_dev: uhfman_set_select_mode failed with error %d", err);
        return -1;
    }
    err = uhfman_set_transmit_power(&pUHFD->uhfmanCtx, txPower);
    if (err != UHFMAN_SET_TRANSMIT_POWER_ERR_SUCCESS) {
        LOG_E("uhfd_embody_dev: uhfman_set_transmit_power failed with error %d", err);
        return -1;
    }

    uint16_t _pc = 0xFFFF;
    uint8_t* _pEPC = NULL;
    size_t _epc_len = 0;
    uint8_t _resp_err = 0;
    // uhfman_tag_mem_bank_t mem_bank = UHFMAN_TAG_MEM_BANK_EPC;
    // uint16_t wordPtr = UHFMAN_TAG_MEM_EPC_WORD_PTR_EPC;
    // uint16_t wordCount = UHFMAN_TAG_MEM_EPC_WORD_COUNT_EPC;
    // const uint8_t access_password[4] = {
    //     0x00, 0x00, 0x00, 0x00
    // };

    // FOR NOW WE ONLY SUPPORT EMBODYING TAG WHICH HAD ZERO PASSWORDS BEFORE SO IT CAN BE DONE ONLY ONCE FOR THAT TAG
    
    assert(TRUE == p_rwlock_reader_lock(pUHFD->pDaRWLock));
    //assert(&pUHFD->da.pDevs[pDev->devno] == pDev); // make sure we are not working with a copy
    //uhfd_dev_t dev = *pDev;
    uhfd_dev_t dev = pUHFD->da.pDevs[devno];
    assert(TRUE == p_rwlock_reader_unlock(pUHFD->pDaRWLock));

    assert((UHFD_DEV_FLAG1_PASSWDS_WRITTEN | UHFD_DEV_FLAG1_PASSWDS_WRITTEN) != (dev.flags1 & (UHFD_DEV_FLAG1_PASSWDS_WRITTEN | UHFD_DEV_FLAG1_PASSWDS_WRITTEN)));
    if (!(dev.flags1 & UHFD_DEV_FLAG1_PASSWDS_WRITTEN)) {
        // Write kill password and access password (by default they are zeroes before doing anything. Should we allow embodying a tag which already has a non-zero password?)
        assert(UHFMAN_TAG_MEM_RESERVED_WORD_PTR_KILL_PASSWD + UHFMAN_TAG_MEM_RESERVED_WORD_COUNT_KILL_PASSWD == UHFMAN_TAG_MEM_RESERVED_WORD_PTR_ACCESS_PASSWD);
        assert(sizeof(dev.kill_passwd) == 2*UHFMAN_TAG_MEM_RESERVED_WORD_COUNT_KILL_PASSWD);
        assert(sizeof(dev.access_passwd) == 2*UHFMAN_TAG_MEM_RESERVED_WORD_COUNT_ACCESS_PASSWD);
        uint8_t passwords[2*(UHFMAN_TAG_MEM_RESERVED_WORD_COUNT_KILL_PASSWD + UHFMAN_TAG_MEM_RESERVED_WORD_COUNT_ACCESS_PASSWD)] = {0};
        memcpy(passwords, dev.kill_passwd, 2*UHFMAN_TAG_MEM_RESERVED_WORD_COUNT_KILL_PASSWD);
        memcpy(passwords + 2*UHFMAN_TAG_MEM_RESERVED_WORD_COUNT_KILL_PASSWD, dev.access_passwd, 2*UHFMAN_TAG_MEM_RESERVED_WORD_COUNT_ACCESS_PASSWD);
        err = uhfman_write_tag_mem(&pUHFD->uhfmanCtx, 
                                    (uint8_t[4]){0x00, 0x00, 0x00, 0x00}, 
                                    UHFMAN_TAG_MEM_BANK_RESERVED, 
                                    UHFMAN_TAG_MEM_RESERVED_WORD_PTR_KILL_PASSWD, 
                                    UHFMAN_TAG_MEM_RESERVED_WORD_COUNT_KILL_PASSWD + UHFMAN_TAG_MEM_RESERVED_WORD_COUNT_ACCESS_PASSWD, 
                                    passwords,
                                    &_pc, &_pEPC, &_epc_len, &_resp_err);
        if (err != UHFMAN_WRITE_TAG_MEM_ERR_SUCCESS) {
            if (err == UHFMAN_WRITE_TAG_MEM_ERR_ERROR_RESPONSE && _resp_err == UHFMAN_TAG_ERR_ACCESS_DENIED) {
                assert(_pEPC != NULL);
                LOG_I_TBC("Access denied for tag (pc = 0x0x%04X, epc_len = %lu, EPC: ", _pc, _epc_len);
                for (size_t i=0; i<_epc_len; i++) {
                    LOG_I_CTBC("%02X ", _pEPC[i]);
                }
                LOG_I_CFIN(")");
                free(_pEPC);
                LOG_E("uhfd_embody_dev: Access denied. Isn't the tag already embodied or is it an alien tag?");
            }
            LOG_E("uhfd_embody_dev: uhfman_write_tag_mem failed with error %d", err);
            assert(TRUE == p_mutex_unlock(pUHFD->pUhfmanCtxMutex));
            assert(!(dev.flags1 & UHFD_DEV_FLAG1_PASSWDS_WRITTEN));
            return -1;
        }
        assert(_pEPC != NULL);
        LOG_I_TBC("Access granted for tag (pc = 0x0x%04X, epc_len = %lu, old EPC: ", _pc, _epc_len);
        for (size_t i=0; i<_epc_len; i++) {
            LOG_I_CTBC("%02X ", _pEPC[i]);
        }
        LOG_I_CTBC(", kill passwd: ");
        for (size_t i=0; i<2*UHFMAN_TAG_MEM_RESERVED_WORD_COUNT_KILL_PASSWD; i++) {
            LOG_I_CTBC("%02X ", dev.kill_passwd[i]);
        }
        LOG_I_CTBC(", access passwd: ");
        for (size_t i=0; i<2*UHFMAN_TAG_MEM_RESERVED_WORD_COUNT_ACCESS_PASSWD; i++) {
            LOG_I_CTBC("%02X ", dev.access_passwd[i]);
        }
        LOG_I_CFIN(")");
        free(_pEPC);
        _pc = 0xFFFF;
        _pEPC = NULL;
        _epc_len = 0;
        _resp_err = 0;
    } else {
        LOG_W("uhfd_embody_dev: Passwords already written for tag (devno = %lu)", dev.devno);
    }
    // Write EPC
    //err = uhfman_write_tag_mem(&pUHFD->uhfmanCtx, access_password, mem_bank, wordPtr, wordCount, pDev->epc, &_pc, &_pEPC, &_epc_len, &_resp_err);
    err = uhfman_write_tag_mem(&pUHFD->uhfmanCtx, 
                                dev.access_passwd, 
                                UHFMAN_TAG_MEM_BANK_EPC, 
                                UHFMAN_TAG_MEM_EPC_WORD_PTR_EPC, 
                                UHFMAN_TAG_MEM_EPC_WORD_COUNT_EPC, 
                                dev.epc,
                                &_pc, &_pEPC, &_epc_len, &_resp_err);
    if (err != UHFMAN_WRITE_TAG_MEM_ERR_SUCCESS) {
        if ((err == UHFMAN_WRITE_TAG_MEM_ERR_ERROR_RESPONSE) && (_resp_err == UHFMAN_TAG_ERR_ACCESS_DENIED)) {
            assert(_pEPC != NULL);
            LOG_I_TBC("Access denied for tag (pc = 0x0x%04X, epc_len = %lu, EPC: ", _pc, _epc_len);
            for (size_t i=0; i<_epc_len; i++) {
                LOG_I_CTBC("%02X ", _pEPC[i]);
            }
            LOG_I_CFIN(")");
            free(_pEPC);
            LOG_E("uhfd_embody_dev: Access denied. Isn't the tag already embodied or is it an alien tag?");
        } 
        LOG_E("uhfd_embody_dev: uhfman_write_tag_mem failed with error %d", err);
        assert(TRUE == p_rwlock_writer_lock(pUHFD->pDaRWLock));
        uhfd_dev_t* pDev = &pUHFD->da.pDevs[devno];
        pDev->flags1 |= UHFD_DEV_FLAG1_PASSWDS_WRITTEN;
        assert((UHFD_DEV_FLAG1_PASSWDS_WRITTEN | ~UHFD_DEV_FLAG1_EPC_WRITTEN) == (pDev->flags1 & (UHFD_DEV_FLAG1_PASSWDS_WRITTEN | UHFD_DEV_FLAG1_EPC_WRITTEN)));
        assert(TRUE == p_rwlock_writer_unlock(pUHFD->pDaRWLock));
        assert(TRUE == p_mutex_unlock(pUHFD->pUhfmanCtxMutex));
        return -1;
    }
    assert(_pEPC != NULL);
    LOG_I_TBC("Access granted for tag (pc = 0x0x%04X, epc_len = %lu, old EPC: ", _pc, _epc_len);
    for (size_t i=0; i<_epc_len; i++) {
        LOG_I_CTBC("%02X ", _pEPC[i]);
    }
    LOG_I_CTBC(", new EPC: ");
    for (size_t i=0; i<_epc_len; i++) {
        LOG_I_CTBC("%02X ", dev.epc[i]);
    }
    LOG_I_CFIN(")");
    free(_pEPC);
    // Set the 'embodied' flag
    assert(TRUE == p_rwlock_writer_lock(pUHFD->pDaRWLock));
    uhfd_dev_t* pDev = &pUHFD->da.pDevs[devno];
    pDev->flags1 = 0x00; // clear flags1
    pDev->flags |= UHFD_DEV_FLAG_EMBODIED;
    assert(TRUE == p_rwlock_writer_unlock(pUHFD->pDaRWLock));
    assert(TRUE == p_mutex_unlock(pUHFD->pUhfmanCtxMutex));
    return 0;
}

int uhfd_measure_dev(uhfd_t* pUHFD, /*uhfd_dev_t* pDev*/unsigned long devno, uhfd_dev_m_t* pMeasurement) {
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
    p_mutex_free(pUHFD->pUhfmanCtxMutex);

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