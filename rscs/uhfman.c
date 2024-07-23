#include "uhfman.h"
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>

// TODO Unify some implementations related to USB with printer.c? Could be as a small shared library for QPS and RSCS

#define UHFMAN_DETACH_KERNEL_DRIVERS_ERR_SUCCESS 0
#define UHFMAN_DETACH_KERNEL_DRIVERS_ERR_NO_DEVICE_WHEN_CHECKING 1
#define UHFMAN_DETACH_KERNEL_DRIVERS_ERR_NOT_SUPPORTED_WHEN_CHECKING 2
#define UHFMAN_DETACH_KERNEL_DRIVERS_ERR_INVALID_PARAM 3
#define UHFMAN_DETACH_KERNEL_DRIVERS_ERR_NO_DEVICE_WHEN_DETACHING 4
#define UHFMAN_DETACH_KERNEL_DRIVERS_ERR_NOT_SUPPORTED_WHEN_DETACHING 5
#define UHFMAN_DETACH_KERNEL_DRIVERS_ERR_OTHER_WHEN_CHECKING 6
#define UHFMAN_DETACH_KERNEL_DRIVERS_ERR_OTHER_WHEN_DETACHING 7
static int uhfman_detach_kernel_drivers(uhfman_ctx_t* pCtx) {
    int rv = libusb_kernel_driver_active(pCtx->handle, UHFMAN_USB_IFACE_IX);

    switch (rv) {
        case 0:
            // No kernel driver active so nothing to do
            return UHFMAN_DETACH_KERNEL_DRIVERS_ERR_SUCCESS;
        case LIBUSB_ERROR_NO_DEVICE:
            return UHFMAN_DETACH_KERNEL_DRIVERS_ERR_NO_DEVICE_WHEN_CHECKING;
        case LIBUSB_ERROR_NOT_SUPPORTED:
            return UHFMAN_DETACH_KERNEL_DRIVERS_ERR_NOT_SUPPORTED_WHEN_CHECKING;
        case 1:
            // Kernel driver active, detach it
            rv = libusb_detach_kernel_driver(pCtx->handle, UHFMAN_USB_IFACE_IX);
            switch (rv) {
                case 0:
                    fprintf(stdout, "Kernel driver detached successfully\n");
                    return UHFMAN_DETACH_KERNEL_DRIVERS_ERR_SUCCESS;
                case LIBUSB_ERROR_INVALID_PARAM:
                    return UHFMAN_DETACH_KERNEL_DRIVERS_ERR_INVALID_PARAM;
                case LIBUSB_ERROR_NO_DEVICE:
                    return UHFMAN_DETACH_KERNEL_DRIVERS_ERR_NO_DEVICE_WHEN_DETACHING;
                case LIBUSB_ERROR_NOT_SUPPORTED:
                    return UHFMAN_DETACH_KERNEL_DRIVERS_ERR_NOT_SUPPORTED_WHEN_DETACHING;
                default:
                    fprintf(stderr, "Unknown error when detaching kernel driver: %d\n", rv);
                    return UHFMAN_DETACH_KERNEL_DRIVERS_ERR_OTHER_WHEN_DETACHING;
            }
        default:
            fprintf(stderr, "Unknown error when checking kernel driver status: %d\n", rv);
            return UHFMAN_DETACH_KERNEL_DRIVERS_ERR_OTHER_WHEN_CHECKING;
        
    }
}

uhfman_err_t uhfman_device_take(uhfman_ctx_t *pCtx_out) {
    libusb_device_handle *handle;
    libusb_context *context = NULL;
    int r;

    // Initialize libusb
    r = libusb_init(&context);
    if (r < 0) {
        fprintf(stderr, "Error initializing libusb: %s\n", libusb_error_name(r));
        return UHFMAN_TAKE_ERR_LIBUSB_INIT;
    }

    // Open the device 
    handle = libusb_open_device_with_vid_pid(context, UHFMAN_VENDOR_ID, UHFMAN_PRODUCT_ID);
    //libusb_open(experiment_global_dev_ptr, &handle);

    if (handle == NULL) {
        if (errno == 0) {
            fprintf(stderr, "Device found, but handle is NULL. Is it a driver issue?\n"); // TODO Return something different than UHFMAN_TAKE_ERR_DEVICE_NOT_FOUND
        }
        else {
            fprintf(stderr, "Error finding USB device\n");
        }
        #if UHFMAN_USE_DEBUG_EXTENSIONS == 1
        uhfman_debug_errno();
        #endif
        libusb_exit(context);
        return UHFMAN_TAKE_ERR_DEVICE_NOT_FOUND;
    }

    // set the context and handle now so that uhfman_detach_kernel_drivers can use them
    pCtx_out->handle = handle; 
    pCtx_out->context = context;

    // Detach kernel drivers if needed
    fprintf(stdout, "Detaching kernel drivers if needed...\n");
    r = uhfman_detach_kernel_drivers(pCtx_out);
    if (r != UHFMAN_DETACH_KERNEL_DRIVERS_ERR_SUCCESS) {
        fprintf(stderr, "Error detaching kernel drivers: %d\n", r);
        libusb_close(handle);
        libusb_exit(context);
        return UHFMAN_TAKE_ERR_INTERFACE_CLAIM;
    }

    // Claim the interface (assuming interface UHFMAN_USB_IFACE_IX)
    r = libusb_claim_interface(handle, UHFMAN_USB_IFACE_IX);
    if (r < 0) {
        fprintf(stderr, "Error claiming interface: %s\n", libusb_error_name(r));
        #if UHFMAN_USE_DEBUG_EXTENSIONS == 1
        uhfman_debug_errno();
        #endif
        libusb_close(handle);
        libusb_exit(context);
        return UHFMAN_TAKE_ERR_INTERFACE_CLAIM;
    }

    return UHFMAN_TAKE_ERR_SUCCESS;
}

void uhfman_device_release(uhfman_ctx_t *pCtx) {
    if (0 != libusb_release_interface(pCtx->handle, 0)) {
        fprintf(stderr, "Error releasing interface\n");
    }
    libusb_close(pCtx->handle);
    libusb_exit(pCtx->context);
}

uhfman_err_t uhfman_get_hardware_version(uhfman_ctx_t* pCtx, char** ppcVersion_out) {
    // TODO implement
    fprintf(stderr, "uhfman_get_hardware_version not implemented\n");
    assert(0);
}

uhfman_err_t uhfman_get_software_version(uhfman_ctx_t* pCtx, char** ppcVersion_out) {
    // TODO implement
    fprintf(stderr, "uhfman_get_hardware_version not implemented\n");
    assert(0);
}

uhfman_err_t uhfman_get_manufacturer(uhfman_ctx_t* pCtx, char** ppcManufacturer_out) {
    // TODO implement
    fprintf(stderr, "uhfman_get_hardware_version not implemented\n");
    assert(0);
}