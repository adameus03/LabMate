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

typedef int uhfman_err_t;

typedef struct {
    libusb_device_handle *handle;
    libusb_context *context;
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

#endif // UHFMAN_COMMON_H