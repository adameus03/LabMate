#include "printer.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
// TODO: Fix this
//#if PRINTER_USE_DEBUG_EXTENSIONS == 1
//#include <string.h> // For strerror
//#endif
#include <string.h> // For strerror
#include <assert.h>


#if PRINTER_MODEL == PRINTER_MODEL_DYMO_LABELWRITER_400
#include "lw400.h"
#endif

#define PRINTER_DETACH_KERNEL_DRIVERS_ERR_SUCCESS 0
#define PRINTER_DETACH_KERNEL_DRIVERS_ERR_NO_DEVICE_WHEN_CHECKING 1
#define PRINTER_DETACH_KERNEL_DRIVERS_ERR_NOT_SUPPORTED_WHEN_CHECKING 2
#define PRINTER_DETACH_KERNEL_DRIVERS_ERR_INVALID_PARAM 3
#define PRINTER_DETACH_KERNEL_DRIVERS_ERR_NO_DEVICE_WHEN_DETACHING 4
#define PRINTER_DETACH_KERNEL_DRIVERS_ERR_NOT_SUPPORTED_WHEN_DETACHING 5
#define PRINTER_DETACH_KERNEL_DRIVERS_ERR_OTHER_WHEN_CHECKING 6
#define PRINTER_DETACH_KERNEL_DRIVERS_ERR_OTHER_WHEN_DETACHING 7
static int printer_detach_kernel_drivers(printer_ctx_t* pCtx) {
    int rv = libusb_kernel_driver_active(pCtx->handle, PRINTER_USB_IFACE_IX);

    switch (rv) {
        case 0:
            // No kernel driver active so nothing to do
            return PRINTER_DETACH_KERNEL_DRIVERS_ERR_SUCCESS;
        case LIBUSB_ERROR_NO_DEVICE:
            return PRINTER_DETACH_KERNEL_DRIVERS_ERR_NO_DEVICE_WHEN_CHECKING;
        case LIBUSB_ERROR_NOT_SUPPORTED:
            return PRINTER_DETACH_KERNEL_DRIVERS_ERR_NOT_SUPPORTED_WHEN_CHECKING;
        case 1:
            // Kernel driver active, detach it
            rv = libusb_detach_kernel_driver(pCtx->handle, PRINTER_USB_IFACE_IX);
            switch (rv) {
                case 0:
                    fprintf(stdout, "Kernel driver detached successfully\n");
                    return PRINTER_DETACH_KERNEL_DRIVERS_ERR_SUCCESS;
                case LIBUSB_ERROR_INVALID_PARAM:
                    return PRINTER_DETACH_KERNEL_DRIVERS_ERR_INVALID_PARAM;
                case LIBUSB_ERROR_NO_DEVICE:
                    return PRINTER_DETACH_KERNEL_DRIVERS_ERR_NO_DEVICE_WHEN_DETACHING;
                case LIBUSB_ERROR_NOT_SUPPORTED:
                    return PRINTER_DETACH_KERNEL_DRIVERS_ERR_NOT_SUPPORTED_WHEN_DETACHING;
                default:
                    fprintf(stderr, "Unknown error when detaching kernel driver: %d\n", rv);
                    return PRINTER_DETACH_KERNEL_DRIVERS_ERR_OTHER_WHEN_DETACHING;
            }
        default:
            fprintf(stderr, "Unknown error when checking kernel driver status: %d\n", rv);
            return PRINTER_DETACH_KERNEL_DRIVERS_ERR_OTHER_WHEN_CHECKING;
        
    }
}

//extern struct libusb_device* experiment_global_dev_ptr; // Experiment global variable inserted into libusb/libusb/core.c (near libusb_open_device_with_vid_pid)

int printer_take(printer_ctx_t *pCtx_out) {
    libusb_device_handle *handle;
    libusb_context *context = NULL;
    int r;

    // Initialize libusb
    r = libusb_init(&context);
    if (r < 0) {
        fprintf(stderr, "Error initializing libusb: %s\n", libusb_error_name(r));
        return PRINTER_TAKE_ERR_LIBUSB_INIT;
    }

    // Open the device 
    // NOW NOT @warning We are using a modified version of libusb_open_device_with_vid_pid which does almost the same thing, but doesn't call libusb_open - this is an experiment aiming to detach kernel drivers before opening the device - i don't know if it will help / work
    handle = libusb_open_device_with_vid_pid(context, PRINTER_VENDOR_ID, PRINTER_PRODUCT_ID);
    //libusb_open(experiment_global_dev_ptr, &handle);

    if (handle == NULL) {
        if (errno == 0) {
            fprintf(stderr, "Device found, but handle is NULL. Is it a driver issue?\n"); // TODO Return something different than PRINTER_TAKE_ERR_DEVICE_NOT_FOUND
        }
        else {
            fprintf(stderr, "Error finding USB device\n");
        }
        #if PRINTER_USE_DEBUG_EXTENSIONS == 1
        printer_debug_errno();
        #endif
        libusb_exit(context);
        return PRINTER_TAKE_ERR_DEVICE_NOT_FOUND;
    }

    // set the context and handle now so that printer_detach_kernel_drivers can use them
    pCtx_out->handle = handle; 
    pCtx_out->context = context;

    // Detach kernel drivers if needed
    fprintf(stdout, "Detaching kernel drivers if needed...\n");
    r = printer_detach_kernel_drivers(pCtx_out);
    if (r != PRINTER_DETACH_KERNEL_DRIVERS_ERR_SUCCESS) {
        fprintf(stderr, "Error detaching kernel drivers: %d\n", r);
        libusb_close(handle);
        libusb_exit(context);
        return PRINTER_TAKE_ERR_INTERFACE_CLAIM;
    }

    // Claim the interface (assuming interface PRINTER_USB_IFACE_IX)
    r = libusb_claim_interface(handle, PRINTER_USB_IFACE_IX);
    if (r < 0) {
        fprintf(stderr, "Error claiming interface: %s\n", libusb_error_name(r));
        #if PRINTER_USE_DEBUG_EXTENSIONS == 1
        printer_debug_errno();
        #endif
        libusb_close(handle);
        libusb_exit(context);
        return PRINTER_TAKE_ERR_INTERFACE_CLAIM;
    }

    return PRINTER_TAKE_ERR_SUCCESS;
}

void printer_release(printer_ctx_t *pCtx) {
    if (0 != libusb_release_interface(pCtx->handle, 0)) {
        fprintf(stderr, "Error releasing interface\n");
    }
    libusb_close(pCtx->handle);
    libusb_exit(pCtx->context);
}

printer_err_t printer_get_revision(printer_ctx_t *pCtx, char *pcRevision_out) {
    #ifdef PRINTER_REVISION_STRING_SUPPORTED
    #if PRINTER_MODEL == PRINTER_MODEL_DYMO_LABELWRITER_400
    typedef uint8_t ___revision_basetype_t;
    ___revision_basetype_t revision[PRINTER_REVISION_STRING_LENGTH];
    printer_err_t err = lw400_esc_V(pCtx, revision);
    if (err != LW400_ESC_V_ERR_SUCCESS) {
        switch (err) {
            case LW400_ESC_V_ERR_SEND_COMMAND:
                return PRINTER_GET_REVISION_ERR_SEND_COMMAND;
            case LW400_ESC_V_ERR_READ_RESPONSE:
                fprintf(stderr, "Error reading response\n");
                return PRINTER_GET_REVISION_ERR_READ_RESPONSE;
            default:
                fprintf(stderr, "Unknown error: %d\n", err);
                assert(0);
                break;
        }
    }
    memcpy(pcRevision_out, revision, sizeof(revision) / sizeof(___revision_basetype_t));
    revision[PRINTER_REVISION_STRING_LENGTH] = '\0';
    return PRINTER_GET_REVISION_ERR_SUCCESS;
    #else
    return PRINTER_GET_REVISION_ERR_UNKNOWN_PRINTER_MODEL;
    #endif // PRINTER_MODEL == PRINTER_MODEL_DYMO_LABELWRITER_400
    #else
    return PRINTER_GET_REVISION_ERR_NOT_SUPPORTED;
    #endif // PRINTER_REVISION_STRING_SUPPORTED
}
