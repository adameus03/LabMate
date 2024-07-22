#ifndef PRINTER_COMMON_H
#define PRINTER_COMMON_H

#include "libusb.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>


#define PRINTER_USE_DEBUG_EXTENSIONS 1

#define PRINTER_ERR_SUCCESS 0
#define PRINTER_ERR_LIBUSB_INIT 1
#define PRINTER_ERR_DEVICE_NOT_FOUND 2
#define PRINTER_ERR_INTERFACE_CLAIM 3
#define PRINTER_ERR_READ_RESPONSE 4
#define PRINTER_ERR_SEND_COMMAND 5
#define PRINTER_ERR_INTERFACE_RELEASE 6
#define PRINTER_ERR_NOT_SUPPORTED 7
#define PRINTER_ERR_UNKNOWN_PRINTER_MODEL 8
#define PRINTER_ERR_CONVERSION_FAILED 9

typedef enum {
    PRINTER_RESOLUTION_300x300_DPI,
    PRINTER_RESOLUTION_203x300_DPI
} printer_resolution_t;

typedef struct {
    libusb_device_handle *handle;
    libusb_context *context;
    struct printer_config { // TODO: Add more settings? (common/model-specific)
        printer_resolution_t resolution;
        int nBytesPerLine;
    } config;
} printer_ctx_t;

typedef int printer_err_t;

#if PRINTER_USE_DEBUG_EXTENSIONS == 1
static inline void printer_debug_errno() {
    int err = errno;
    printf("errno: %d\n", err);
    char* errStr = strerror(err);
    printf("strerror(errno): %s\n", errStr);
}

#endif

#endif // PRINTER_COMMON_H
