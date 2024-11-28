/*
    bash/client software --- database
         |
    (filesystem)
         |
       main (FUSE)
         |
       uhfd <<< we are here
        |
      uhfman
        |
     ypdr200
*/

/*
    Device files will be stored in /custom/uhfX, where X is the device number.
    Per-device driver control files will be stored in /custom/uhfX/driver.
    Global driver control files will be stored in /custom/uhfd

    uhfX
        epc [rw]
        access_passwd [w]
        kill_passwd [w]
        flags [rw]
        rssi [r]        }  combine to space-separated string?
        read_rate [r]   }
        driver
            //flags [w] - for deletion
            //request [w] - 'd'/'e'/'m' ((d)elete, (e)mbody, (m)easure)
            //ready [r] - 0/1
            delete [w] - write flags to delete
            embody [w] - write 1
            measure [w] - write timeout value to trigger measurement stored in rssi and read_rate
    uhfd
        //max_index [internal]
        //request - [w] only 'c' for create
        //ready - [r] 0/1
        //mkdev [r] - returns the next available device number
        
        sid [r] // obtains driver-generated session id for requests (store last N in memory of the driver program to handle max N concurrent tasks)
        mkdev [w] // Trigger new device directory setup by driver and also the driver would create a file containing output device number at result/$sid/value
            The default values are:
                epc: 12B (zeroes)
                access_passwd: 4B (zeroes)
                kill_passwd: 4B (zeroes)
                flags: 0x00 (not ignored, not embodied)
        result 
            $sid
                value [r] // the device number
                fin [w] - write 1 // Delete this session id from the driver's memory (finalize the session) 


*/

// If antenna switching is required, it supposed to be done by separate software

#include <stdint.h>
//#include <mtwister.h>
#include "mtwister.h"
#include "uhfman_common.h"
#include <plibsys.h>

#define UHFD_DEV_FLAG_IGNORED 0x01
#define UHFD_DEV_FLAG_EMBODIED 0x02
#define UHFD_DEV_FLAG_DELETED 0x04

#define UHFD_DEV_FLAG1_PASSWDS_WRITTEN 0x01
#define UHFD_DEV_FLAG1_EPC_WRITTEN 0x02

#define UHFD_NUM_DEVS_INITIAL 16
#define UHFD_NUM_DEVS_SCALING_FACTOR 1.5

#define UHFD_STATE_FLAG_INITIALIZED 0x01

// #define UHFD_LOCK_FLAG_STATE_MUTEX 0x01
// #define UHFD_LOCK_FLAG_DEVCSIZE_MUTEX 0x02
// #define UHFD_LOCK_FLAG_DEV_RWLOCK_READ 0x04
// #define UHFD_LOCK_FLAG_DEV_RWLOCK_WRITE 0x08

#define UHFD_EPC_LENGTH UHFMAN_TAG_EPC_STANDARD_LENGTH

typedef struct {
    uint8_t rssi;
    uint8_t read_rate;
} uhfd_dev_m_t;

// typedef enum {
//     UHFD_DEV_EMB_STATE_READY = 0x00,
//     UHFD_DEV_EMB_STATE
//     UHFD_DEV_EMB_STATE_PASSWDS_WRITTEN = 0x01,
//     UHFD_DEV_EMB_STATE_EPC_WRITTEN = 0x02,
// } uhfd_dev_emb_state_t;

typedef struct {
    uint8_t epc[UHFD_EPC_LENGTH];
    uint8_t access_passwd[4];
    uint8_t kill_passwd[4];
    uint8_t flags;
    uint8_t flags1; // internal flags for handling partial write
    //uint8_t emb_state; // internal state for handling partial progress during embodiment
    uhfd_dev_m_t measurement;
    unsigned long devno;
} uhfd_dev_t;

typedef struct {
    unsigned long num_devs;
    unsigned long num_devs_flex_max;
    uhfd_dev_t* pDevs;
} uhfd_dev_array_t;

// TODO refactor internal fields to a separate struct?
typedef struct {
    MTRand mtrand; // uhfd internal
    uint8_t state_flags; // uhfd internal note: Please access atomically (pStateMutex and pInitDeinitMutex are obsoleted)
    //unsigned long num_devs; // [obsolete] Please lock using pDevCSizeMutex. Same for num_devs_flex_max
    //unsigned long num_devs_flex_max; // [obsolete] When this number is reached, the memory block for storing uhfd_dev_t structures will be enlarged by a factor of UHFD_NUM_DEVS_SCALING_FACTOR using realloc
    
    uhfd_dev_array_t da; // uhfd internal note: Please lock/unlock using pDaRWLock (pDevCSizeMutex is obsoleted)

    //uhfd_dev_t* pDevs; // [obsolete] Please access via atomic operations on individual elements (pDevRWLock is obsoleted)
    //PMutex* pInitDeinitMutex; // [obsolete] The aim is to ensure that the initialization and deinitialization functions are not called concurrently
    //PMutex* pCreateDevMutex; // [obsolete] The aim is to ensure integrity of num_devs and num_devs_flex_max
    //PRWLock* pDevRWLock; // [obsolete] Single coarse lock for the whole array stored at pDevs
                        // If neccessary in the future, we can implement fine-grained locking. Think about spinlocks as well?
    //PMutex* pStateMutex; // [obsolete] For state_flags and  handling state transitions
    //PMutex* pDevCSizeMutex; // [obsolete] For accessing num_devs and num_devs_flex_max
    //PRWLock* pDevRWLock; // [obsolete] For device read/write operations. Single coarse lock for the whole array stored at pDevs. If neccessary in the future, we can implement fine-grained locking. Think about spinlocks as well (not sure about them actually). For now reallocs are handled using this lock as well (together with pDevCSizeMutex). ~~We can't use atomic operations instead, because we need to lock the entire array for realloc.~~
    
    PRWLock* pDaRWLock; // uhfd internal. For device array read/write operations. Single coarse lock for the whole `da` struct field. If neccessary in the future, we can implement fine-grained locking for da.pDev. Think about spinlocks as well (not sure about them actually). For now da.pDev reallocs are handled using this lock as well (together with pDevCSizeMutex). We can't use atomic operations instead, because they work on memory ranges which are too small on popular hardware platforms.
    PMutex* pUhfmanCtxMutex; // uhfd internal. For uhfman_ctx_t operations. 
    //PMutex* pDevsReallocMutex; // [obsolete] For reallocs of the pDevs array

    uhfman_ctx_t uhfmanCtx; // refactor to uhfman_ctx ?
} uhfd_t;

typedef uint8_t uhfd_combined_lock_flags_t;

// void uhfd_combined_lock(uhfd_t* pUHFD, uhfd_combined_lock_flags_t flags); // This and uhfd_unlock would need adjustments if fine-grained locking is implemented

// void uhfd_combined_unlock(uhfd_t* pUHFD, uhfd_combined_lock_flags_t flags);

int uhfd_init(uhfd_t* pUHFD_out);

// int uhfd_create_dev(uhfd_t* pUHFD, uhfd_dev_t* pDev_out);
int uhfd_create_dev(uhfd_t* pUHFD, unsigned long* pDevNum_out);

// Utility dev getter function. Avoids forcing the user to read lock a uhfd internal rwlock
int uhfd_get_dev(uhfd_t* pUHFD, unsigned long devno, uhfd_dev_t* pDev_out);

// Utility dev setter function. Avoids forcing the user to write lock a uhfd internal rwlock
int uhfd_set_dev(uhfd_t* pUHFD, unsigned long devno, uhfd_dev_t dev);

int uhfd_get_num_devs(uhfd_t* pUHFD, unsigned long* pNumDevs_out);

// [obsolete] Can't replace with atomic operations because a rwlock (at pDaRWLock) is already used for synchronizing access to the whole array including its memembers' fields
// int uhfd_get_dev_flags_synchronized(uhfd_t* pUHFD, unsigned long devno, uint8_t* pFlags_out);

//should we even bother to kill the tag? (it may not be neccessary!)
/* 1 indicates that the underlying tag should be killed */
#define UHFD_DELETE_DEV_FLAG_KILL (1U << 0)
/* 1 indicates that the dev will be removed from the directory, meanwhile
   0 indicates that the dev will still remain in the directory, though having the UHFD_DEV_FLAG_IGNORED flag set */
#define UHFD_DELETE_DEV_FLAG_REMOVE (1U << 1)

int uhfd_delete_dev(uhfd_t* pUHFD, unsigned long devno, uint8_t flags);

int uhfd_embody_dev(uhfd_t* pUHFD, unsigned long devno);

int uhfd_measure_dev(uhfd_t* pUHFD, unsigned long devno, unsigned long timeout_us, float tx_power); // DONE remove pMeasurement (but not its type)

int uhfd_quick_measure_dev_rssi(uhfd_t* pUHFD, unsigned long devno, float tx_power);

int uhfd_deinit(uhfd_t* pUHFD);