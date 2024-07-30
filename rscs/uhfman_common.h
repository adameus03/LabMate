#ifndef UHFMAN_COMMON_H
#define UHFMAN_COMMON_H

#include "libusb.h"

#define UHFMAN_USE_DEBUG_EXTENSIONS 1

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

#define UHFMAN_ERR_OTHER 100

typedef int uhfman_err_t;

typedef struct {
    /* For libusb */
    libusb_device_handle *handle;
    libusb_context *context;
    /* For serial port emulation */
    int fd;
    /* Other config */
    struct uhfman_config {
        // Nothing here for now
    } config;
} uhfman_ctx_t;

#if UHFMAN_USE_DEBUG_EXTENSIONS == 1
#include <stdio.h>
#include <errno.h>
#include <string.h>
static inline void uhfman_debug_errno() {
    int err = errno;
    printf("errno: %d\n", err);
    char* errStr = strerror(err);
    printf("strerror(errno): %s\n", errStr);
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