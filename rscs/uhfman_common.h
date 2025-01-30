#ifndef UHFMAN_COMMON_H
#define UHFMAN_COMMON_H

#include "libusb.h"
#include <poll.h>
#include "log.h"

#define UHFMAN_USE_DEBUG_EXTENSIONS 1

#define UHFMAN_GREEDY_MODE 0

#define UHFMAN_TAG_EPC_STANDARD_LENGTH 12

#define UHFMAN_ERR_SUCCESS 0
#define UHFMAN_ERR_LIBUSB_INIT 1
#define UHFMAN_ERR_DEVICE_NOT_FOUND 2
#define UHFMAN_ERR_INTERFACE_CLAIM 3
#define UHFMAN_ERR_READ_RESPONSE 4
#define UHFMAN_ERR_SEND_COMMAND 5
#define UHFMAN_ERR_INTERFACE_RELEASE 6
#define UHFMAN_ERR_NOT_SUPPORTED 7
#define UHFMAN_ERR_UNKNOWN_DEVICE_MODEL 8
#define UHFMAN_ERR_BRIDGE_INIT_FAIL 9
#define UHFMAN_ERR_ERROR_RESPONSE 10
#define UHFMAN_ERR_READ_NOTIFICATION 11
#define UHFMAN_ERR_UNEXPECTED_FRAME_TYPE 12
#define UHFMAN_ERR_UNKNOWN 13

#define UHFMAN_ERR_OTHER 100

typedef int uhfman_err_t;

typedef enum {
    // Before any operation
    UHFMAN_SELECT_MODE_ALWAYS = 0x00,
    // Never
    UHFMAN_SELECT_MODE_NEVER = 0x01,
    // Before read, write, lock, kill operations
    UHFMAN_SELECT_MODE_RWLK = 0x02,
    // Unknown
    UHFMAN_SELECT_MODE_UNKNOWN = 0xFF
} uhfman_select_mode_t;

typedef enum {
    UHFMAN_QUERY_SEL_ALL = 0x00,
    UHFMAN_QUERY_SEL_ALL1 = 0x01,
    UHFMAN_QUERY_SEL_NOT_SL = 0x02,
    UHFMAN_QUERY_SEL_SL = 0x03,
    UHFMAN_QUERY_SEL_UNKNOWN = 0xFF
} uhfman_query_sel_t;

typedef enum {
    UHFMAN_QUERY_SESSION_S0 = 0x00,
    UHFMAN_QUERY_SESSION_S1 = 0x01,
    UHFMAN_QUERY_SESSION_S2 = 0x02,
    UHFMAN_QUERY_SESSION_S3 = 0x03,
    UHFMAN_QUERY_SESSION_UNKNOWN = 0xFF
} uhfman_query_session_t;

typedef enum {
    UHFMAN_QUERY_TARGET_A = 0x00,
    UHFMAN_QUERY_TARGET_B = 0x01,
    UHFMAN_QUERY_TARGET_UNKNOWN = 0xFF
} uhfman_query_target_t;

#define UHFMAN_CTX_CONFIG_FLAG_SELECT_INITIALIZED (1U << 0)
#define UHFMAN_CTX_CONFIG_FLAG_QUERY_INITIALIZED (1U << 1)
#define UHFMAN_CTX_CONFIG_FLAG_SELECT_MODE_INITIALIZED (1U << 2)
#define UHFMAN_CTX_CONFIG_FLAG_TX_POWER_INITIALIZED (1U << 3)
#define UHFMAN_CTX_CONFIG_FLAG_IS_MPOLL_BUSY (1U << 4)
typedef struct {
    /* For libusb */
    libusb_device_handle *handle;
    libusb_context *context;
    /* For serial port emulation */
    int fd;
    struct pollfd pollin_fd;
    struct pollfd pollout_fd;
    int pollin_timeout;
    int pollout_timeout;
    
    struct uhfman_config {
        struct {
            uint8_t target;
            uint8_t action; 
            uint8_t memBank;
            uint32_t ptr;
            uint8_t maskLen; 
            uint8_t truncate; 
            uint8_t* pMask;
        } select_params;
        uhfman_select_mode_t select_mode;
        struct {
            uhfman_query_sel_t sel;
            uhfman_query_session_t session;
            uhfman_query_target_t target;
            uint8_t q;            
        } query_params;
        float txPower;
        uint8_t flags; // config flags
    } _config;
} uhfman_ctx_t;

#if UHFMAN_USE_DEBUG_EXTENSIONS == 1
#include <stdio.h>
#include <errno.h>
#include <string.h>
static inline void uhfman_debug_errno() {
    int err = errno;
    LOG_E("errno: %d\n", err);
    char* errStr = strerror(err);
    LOG_E("strerror(errno): %s\n", errStr);
}
#endif

#define UHFMAN_DEVICE_MODEL_YDPR200 1
#define UHFMAN_DEVICE_MODEL UHFMAN_DEVICE_MODEL_YDPR200

#define UHFMAN_DEVICE_CONNECTION_TYPE_UART 0
#define UHFMAN_DEVICE_CONNECTION_TYPE_CH340_USB 1
#define UHFMAN_DEVICE_CONNECTION_TYPE UHFMAN_DEVICE_CONNECTION_TYPE_CH340_USB

#if UHFMAN_DEVICE_CONNECTION_TYPE == UHFMAN_DEVICE_CONNECTION_TYPE_CH340_USB
#define UHFMAN_CH340_USE_KERNEL_DRIVER 1
#if UHFMAN_CH340_USE_KERNEL_DRIVER == 1
    #define UHFMAN_CH340_PORT_NAME "/dev/ttyUSB0"
#endif
#endif

#if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    
#else
#error "Unknown device model for uhfman"
#endif

#if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    #if UHFMAN_DEVICE_CONNECTION_TYPE == UHFMAN_DEVICE_CONNECTION_TYPE_CH340_USB && UHFMAN_CH340_USE_KERNEL_DRIVER == 1
        #define YPDR200_INTERFACE_TYPE_LIBUSB 0
        #define YPDR200_INTERFACE_TYPE_SERIAL 1
        #define YPDR200_INTERFACE_TYPE YPDR200_INTERFACE_TYPE_SERIAL
    #endif
    #include "ypdr200.h"
#endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200

#endif // UHFMAN_COMMON_H