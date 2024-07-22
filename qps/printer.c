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

#include "qr_data_adapter.h"

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
    pcRevision_out[PRINTER_REVISION_STRING_LENGTH] = '\0';
    return PRINTER_GET_REVISION_ERR_SUCCESS;
    #else
    return PRINTER_GET_REVISION_ERR_UNKNOWN_PRINTER_MODEL;
    #endif // PRINTER_MODEL == PRINTER_MODEL_DYMO_LABELWRITER_400
    #else
    return PRINTER_GET_REVISION_ERR_NOT_SUPPORTED;
    #endif // PRINTER_REVISION_STRING_SUPPORTED
}

printer_err_t printer_setup(printer_ctx_t *pCtx) {
    #if PRINTER_MODEL == PRINTER_MODEL_DYMO_LABELWRITER_400
        assert(PRINTER_LABEL_WIDTH_MM <= PRINTER_MAX_LABEL_WIDTH_MM);
        uint16_t nDotsPerLine = (uint16_t)(((double)PRINTER_LABEL_WIDTH_MM) / 25.4 * 300); //TODO replace magic number 300
        assert(nDotsPerLine == 153U); // TODO: Remove after testing
        nDotsPerLine = 144U; //Set constant for now to test printing with eppendorfs size labels (4(pixel expansion) * 33(QR width or height) + 12(padding) = 144) //TODO remove this assignment and make it work for different label sizes
        #if PRINTER_LABEL_IS_SUBDIVIDED == 1
            // Diptych
            //nDotsPerLine <<= 1;
            nDotsPerLine = ((double)PRINTER_LABEL_WIDTH_MM * 2.0) / 25.4 * 300; // TODO replace magic number 300
            //int separationWidth = (PRINTER_LABEL_WIDTH_MM - (PRINTER_LABEL_MARGIN_LEFT_UM/1000)) / 25.4 * 300 - labelGrayscaleDataWidth;
        #endif //PRINTER_LABEL_IS_SUBDIVIDED == 1
        uint8_t nBytesPerLine = (uint8_t)(nDotsPerLine >> 3);
        if (nDotsPerLine % 8 != 0) {
            nBytesPerLine++;
            fprintf(stdout, "WARNING: nBytesPerLine is not a multiple of 8, padding with an extra byte(s)\n");
        }

        printer_err_t err = lw400_esc_D(pCtx, nBytesPerLine);
        if (err != LW400_ESC_D_ERR_SUCCESS) {
            switch (err) {
                case LW400_ESC_D_ERR_SEND_COMMAND:
                    return PRINTER_SETUP_ERR_SEND_COMMAND;
                default:
                    fprintf(stderr, "lw400_esc_D returned unknown error code: %d\n", err);
                    assert(0);
                    break;
            }
        }
        uint16_t nDotsTab = (uint16_t)(((double)PRINTER_LABEL_MARGIN_LEFT_UM) / 25400 * 300); // TODO replace magic number 300
        assert(nDotsTab == /*23*/0); // TODO remove this assertion
        uint8_t nBytesTab = (uint8_t)(nDotsTab >> 3);
        if (nDotsTab % 8 != 0) {
            nBytesTab++;
        }
        assert(nBytesTab == /*3*/0); // TODO remove this assertion
        err = lw400_esc_B(pCtx, nBytesTab);
        if (err != LW400_ESC_B_ERR_SUCCESS) {
            switch (err) {
                case LW400_ESC_B_ERR_SEND_COMMAND:
                    return PRINTER_SETUP_ERR_SEND_COMMAND; // TODO return more specific error codes to indicate when the error happened - e.g. here it happened while trying to set the tab size so we could return a new error code PRINTER_SETUP_ERR_SET_TAB_SIZE_SEND_COMMAND_FAILED or in the default case we could return PRINTER_SETUP_ERR_SET_TAB_SIZE_UNKNOWN_ERROR. It applies also to the other switch statements in this function.
                default:
                    fprintf(stderr, "lw400_esc_B returned unknown error code: %d\n", err);
                    assert(0);
                    break;
            }
        }

        // Set label length as well (not too short!) If too long, the printer's motor should stop just at the IR detection hole (between labels). It should, right? It should? :D
        int16_t labelLength = (int16_t)(1.5 * ((double)PRINTER_LABEL_HEIGHT_MM) / 25.4 * 300); // TODO replace magic numbers 300 and 1.5
        err = lw400_esc_L(pCtx, labelLength);
        if (err != LW400_ESC_L_ERR_SUCCESS) {
            switch (err) {
                case LW400_ESC_L_ERR_SEND_COMMAND:
                    return PRINTER_SETUP_ERR_SEND_COMMAND;
                default:
                    fprintf(stderr, "lw400_esc_L returned unknown error code: %d\n", err);
                    assert(0);
                    break;
            }
        }

        // Check status for testing
        // TODO Do that status check before printing, and if not ready, either wait until ready or return a not ready/error status
        err = lw400_esc_A(pCtx, NULL);
        if (err != LW400_ESC_A_ERR_SUCCESS) {
            switch (err) {
                case LW400_ESC_A_ERR_SEND_COMMAND:
                    return PRINTER_SETUP_ERR_SEND_COMMAND;
                case LW400_ESC_A_ERR_READ_RESPONSE:
                    fprintf(stderr, "Error reading response\n");
                    return PRINTER_SETUP_ERR_READ_RESPONSE;
                default:
                    fprintf(stderr, "lw400_esc_A returned unknown error code: %d\n", err);
                    assert(0);
                    break;
            }
        }
        // TODO Return an error if the printer is not ready for printing (response to <esc> A is not 0x03)

        pCtx->config.nBytesPerLine = (int)nBytesPerLine;
        return PRINTER_SETUP_ERR_SUCCESS;
    #else
    return PRINTER_PRINT_LABEL_ERR_UNKNOWN_PRINTER_MODEL;
    #endif
}

#if PRINTER_LABEL_IS_SUBDIVIDED == 0
printer_err_t printer_print(printer_ctx_t *pCtx, const uint8_t* pLabelGrayscaleData, const int labelGrayscaleDataWidth, const int labelGrayscaleDataHeight) {
#else
printer_err_t printer_print(printer_ctx_t *pCtx, const uint8_t* pLabelGrayscaleDataDiptychLeft, const uint8_t* pLabelGrayscaleDataDiptychRight, const int labelGrayscaleDataWidth, const int labelGrayscaleDataHeight) {
#endif // PRINTER_LABEL_IS_SUBDIVIDED == 0
    #if PRINTER_MODEL == PRINTER_MODEL_DYMO_LABELWRITER_400
        // TODO complete this
        // TODO test data conversion before completing this (don't actually print while doing the first-time test?)
        // TODO include a printer status check before printing (and after printing as well?)

        // Prepare data buffer in format that the printer will understand
        uint8_t* pPrinterDbuf = NULL;
        int nPrinterDbufSize = 0;

        #if PRINTER_LABEL_IS_SUBDIVIDED == 0
        int rv = qda_grayscale_to_dlw400u8buf(pLabelGrayscaleData, labelGrayscaleDataWidth, labelGrayscaleDataHeight, &pPrinterDbuf, &nPrinterDbufSize);
        if (rv != QDA_GRAYSCALE_TO_DLW400U8BUF_ERR_SUCCESS) {
            switch (rv) {
                case QDA_GRAYSCALE_TO_DLW400U8BUF_ERR_MALLOC:
                    return PRINTER_PRINT_ERR_CONVERSION_FAILED;
                default:
                    fprintf(stderr, "qda_grayscale_to_dlw400u8buf returned unknown error code: %d\n", rv);
                    assert(0);
                    break;
            }
        }
        assert(nPrinterDbufSize == labelGrayscaleDataWidth / 8 * labelGrayscaleDataHeight);
        #else
        int separationWidth = (PRINTER_LABEL_WIDTH_MM - (PRINTER_LABEL_MARGIN_LEFT_UM/1000)) / 25.4 * 300 - labelGrayscaleDataWidth; // separation dots between two parts of the diptych print regions
        assert(separationWidth < labelGrayscaleDataWidth); // violation of this assertion would be suspiscious - we can expect the separation width be smaller than the diptych part print width
        assert(separationWidth >= 0);
        int rv = qda_grayscale_diptych_to_dlw400u8buf(pLabelGrayscaleDataDiptychLeft, pLabelGrayscaleDataDiptychRight, labelGrayscaleDataWidth, labelGrayscaleDataHeight, separationWidth, &pPrinterDbuf, &nPrinterDbufSize);
        #endif // PRINTER_LABEL_IS_SUBDIVIDED == 0

        #if PRINTER_LABEL_IS_SUBDIVIDED == 0
        int nBytesPerLine = labelGrayscaleDataWidth / 8;
        #else
        int nBytesPerLine = (labelGrayscaleDataWidth * 2 + separationWidth) / 8;
        #endif // PRINTER_LABEL_IS_SUBDIVIDED == 0
        //assert(nBytesPerLine == pCtx->config.nBytesPerLine); // @note Assertion removed, because extra padding bytes are used for diptych printing (lazy approach)
        #if PRINTER_LABEL_IS_SUBDIVIDED == 0
        assert(nBytesPerLine == 18); // 144/8 = 18 //TODO: Handle different print widths
        #else
        assert(nBytesPerLine == 36 + separationWidth);
        #endif // PRINTER_LABEL_IS_SUBDIVIDED == 1
        // assert(PRINTER_RESOLUTION_300x300_DPI == pCtx->config.resolution); // TODO: Handle different resolutions // assertion removed because of 'lazy' diptych printing
        // Send print cmd+data stream
        uint8_t* pHeadData = pPrinterDbuf; // pointer to current data line
        printer_err_t err;
        fprintf(stdout, "-- Printing data lines --\n");
        for (int i = 0; i < labelGrayscaleDataHeight; i++, pHeadData+=nBytesPerLine) {
            // print current data line
            err = lw400_syn(pCtx, pHeadData, nBytesPerLine);
            if (err != LW400_SYN_ERR_SUCCESS) {
                switch (err) {
                    case LW400_SYN_ERR_SEND_COMMAND:
                        fprintf(stdout, "NOTICE: THE PRINTING PROCESS HAS BEEN INTERRUPTED, YOU SHOULD TAKE CARE OF THE PRINTER\n"); // TODO: Handle this. Either retry or reset the printer and try to complete the remaining lines? (or form feed incomplete label?) Or maybe the usb connection is lost?
                        return PRINTER_PRINT_ERR_SEND_COMMAND;
                    default:
                        fprintf(stderr, "lw400_syn returned unknown error code: %d\n", err);
                        assert(0);
                        break;
                }
            }
            fprintf(stdout, "."); // TODO remove this fprintf after testing
        }
        fprintf(stdout, "\n-- Printing lines completed --\n");

        // Send form feed command so that the label can be torn off
        err = lw400_esc_E(pCtx);
        if (err != LW400_ESC_E_ERR_SUCCESS) {
            switch (err) {
                case LW400_ESC_E_ERR_SEND_COMMAND:
                    return PRINTER_PRINT_ERR_SEND_COMMAND;
                default:
                    fprintf(stderr, "lw400_esc_E returned unknown error code: %d\n", err);
                    assert(0);
                    break;
            }
        }
        return PRINTER_PRINT_ERR_SUCCESS;
    #else
    return PRINTER_PRINT_ERR_UNKNOWN_PRINTER_MODEL;
    #endif
}
