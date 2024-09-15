/*
    bash/client software --- database
         |
    (filesystem)
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
            measure [w] - write 1 to trigger measurement stored in rssi and read_rate
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

typedef struct {
    uint8_t rssi;
    uint8_t read_rate;
} uhfd_dev_m_t;

#define UHFD_DEV_FLAG_IGNORED 0x01
#define UHFD_DEV_FLAG_EMBODIED 0x02
#define UHFD_DEV_FLAG_DELETED 0x04
typedef struct {
    uint8_t epc[12];
    uint8_t access_passwd[4];
    uint8_t kill_passwd[4];
    uint8_t flags;
    uhfd_dev_m_t measurement;
    unsigned long devno;
} uhfd_dev_t;

#define UHFD_NUM_DEVS_INITIAL 16
#define UHFD_NUM_DEVS_SCALING_FACTOR 1.5

#define UHFD_STATE_FLAG_INITIALIZED 0x01
typedef struct {
    MTRand mtrand;
    uint8_t state_flags;
    unsigned long num_devs;
    unsigned long num_devs_flex_max; // When this number is reached, the memory block for storing uhfd_dev_t structures will be enlarged by a factor of UHFD_NUM_DEVS_SCALING_FACTOR using realloc
    uhfd_dev_t* pDevs;
    PMutex* pInitDeinitMutex;
    PMutex* pCreateDevMutex;
    uhfman_ctx_t uhfmanCtx;
} uhfd_t;

int uhfd_init(uhfd_t* pUHFD_out);

//int uhfd_create_dev(uhfd_t* pUHFD, uhfd_dev_t* pDev_out);
int uhfd_create_dev(uhfd_t* pUHFD, unsigned long* pDevNum_out);

//should we even bother to kill the tag? (it may not be neccessary!)
/* 1 indicates that the underlying tag should be killed */
#define UHFD_DELETE_DEV_FLAG_KILL 0x01
/* 1 indicates that the dev will be removed from the directory, meanwhile
   0 indicates that the dev will still remain in the directory, though having the UHFD_DEV_FLAG_IGNORED flag set */
#define UHFD_DELETE_DEV_FLAG_REMOVE 0x02

int uhfd_delete_dev(uhfd_t* pUHFD, uhfd_dev_t* pDev, uint8_t flags);

int uhfd_embody_dev(uhfd_t* pUHFD, uhfd_dev_t* pDev);

int uhfd_measure_dev(uhfd_t* pUHFD, uhfd_dev_t* pDev, uhfd_dev_m_t* pMeasurement);

int uhfd_deinit(uhfd_t* pUHFD);