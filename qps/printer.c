#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
// TODO: Fix this
//#if PRINTER_USE_DEBUG_EXTENSIONS == 1
//#include <string.h> // For strerror
//#endif
#include <string.h> // For strerror

#include "printer.h"

// -- DYMO LW 550 --
//#define DYMO_VENDOR_ID 0x0922
//#define DYMO_PRODUCT_ID 0x0028

// -- DYMO LW 400 --
#define DYMO_VENDOR_ID 0x0922
#define DYMO_PRODUCT_ID 0x0019

#define DYMO_ENDPONT_ADDR_IN 0x82U
#define DYMO_ENDPONT_ADDR_OUT 0x02U

//#define DYMO_VENDOR_ID 0x046d
//#define DYMO_PRODUCT_ID 0xc534


// DONE: Separate libusb and device initialization into separate functions - most functions should be called assuming the device is already initialized
// TODO: Abstract away libusb_bulk_transfer calls into a function that takes a direction, data, and length, so the code is more library-agnostic

#if PRINTER_USE_DEBUG_EXTENSIONS == 1
static void __printer_debug_errno() {
    int err = errno;
    printf("errno: %d\n", err);
    char* errStr = strerror(err);
    printf("strerror(errno): %s\n", errStr);
}
#endif

int printer_esc_v(printer_ctx_t* pCtx) {
    libusb_device_handle* handle = pCtx->handle;

    unsigned char data_out[] = {0x1B, 0x56};
    unsigned char data_in[64];
    int actual_length;

    // Send the command to the printer
    //int r = libusb_bulk_transfer(handle, (1 | LIBUSB_ENDPOINT_OUT), data_out, sizeof(data_out), &actual_length, 0);
    int r = libusb_bulk_transfer(handle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), &actual_length, 0);
    if (r == 0 && actual_length == sizeof(data_out)) {
        printf("Command sent successfully\n");
    } else {
        fprintf(stderr, "Error sending command: %s\n", libusb_error_name(r));
        return PRINTER_ESC_V_ERR_SEND_COMMAND;
    }

    // Read the response from the printer
    //r = libusb_bulk_transfer(handle, (1 | LIBUSB_ENDPOINT_IN), data_in, sizeof(data_in), &actual_length, 5000); // 5 second timeout
    r = libusb_bulk_transfer(handle, DYMO_ENDPONT_ADDR_IN, data_in, sizeof(data_in), &actual_length, 5000); // 5 second timeout
    if (r == 0) {
        printf("Response received:\n");
        for (int i = 0; i < actual_length; i++) {
            printf("%02X ", data_in[i]);
        }
        printf("\n");
    } else {
        fprintf(stderr, "Error reading response: %s\n", libusb_error_name(r));
        return PRINTER_ESC_V_ERR_READ_RESPONSE;
    }

    return PRINTER_ESC_V_ERR_SUCCESS;
}

int printer_esc_d(printer_ctx_t* pCtx, uint8_t *data, int width, int height) {return 0;
    libusb_device_handle *handle = pCtx->handle;

    unsigned char data_out[width * height + 8];
    unsigned char data_in[64];
    int actual_length;

    // Prepare the data to send to the printer
    data_out[0] = 0x1B;
    data_out[1] = 0x44;
    data_out[2] = 0x01;
    data_out[3] = 0x02;
    int* pWidth = (int*)(data_out + 4);
    int* pHeight = (int*)(data_out + 8);
    *pWidth = width;
    *pHeight = height;

    for (int i = 0; i < width * height; i++) {
        data_out[12 + i] = data[i];
    }

    // Send the command to the printer
    int r = libusb_bulk_transfer(handle, (1 | LIBUSB_ENDPOINT_OUT), data_out, sizeof(data_out), &actual_length, 0);
    if (r == 0 && actual_length == sizeof(data_out)) {
        printf("Command sent successfully\n");
    } else {
        fprintf(stderr, "Error sending command: %s\n", libusb_error_name(r));
        #if PRINTER_USE_DEBUG_EXTENSIONS == 1
        __printer_debug_errno();
        #endif
        return PRINTER_ESC_D_ERR_SEND_COMMAND;
    }

    return PRINTER_ESC_D_ERR_SUCCESS;
}

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

extern struct libusb_device* experiment_global_dev_ptr; // Experiment global variable inserted into libusb/libusb/core.c (near libusb_open_device_with_vid_pid)

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
    handle = libusb_open_device_with_vid_pid(context, DYMO_VENDOR_ID, DYMO_PRODUCT_ID);
    //libusb_open(experiment_global_dev_ptr, &handle);

    if (handle == NULL) {
        if (errno == 0) {
            fprintf(stderr, "Device found, but handle is NULL. Is it a driver issue?\n"); // TODO Return something different than PRINTER_TAKE_ERR_DEVICE_NOT_FOUND
        }
        else {
            fprintf(stderr, "Error finding USB device\n");
        }
        #if PRINTER_USE_DEBUG_EXTENSIONS == 1
        __printer_debug_errno();
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
        __printer_debug_errno();
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
