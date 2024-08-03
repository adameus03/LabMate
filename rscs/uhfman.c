#include "uhfman.h"
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

#include "ch340.h"

// TODO Unify some implementations related to USB with printer.c? Could be as a small shared library for QPS and RSCS
// TODO Reattach kernel drivers when exiting program? Or not? What about qps as well? (Though the programs should never really exit)

#if UHFMAN_DEVICE_CONNECTION_TYPE == UHFMAN_DEVICE_CONNECTION_TYPE_UART
    #error "UHFMAN_DEVICE_CONNECTION_TYPE_UART not supported for now"
#elif UHFMAN_DEVICE_CONNECTION_TYPE == UHFMAN_DEVICE_CONNECTION_TYPE_CH340_USB
    #include <libusb.h>
    #define UHFMAN_CH340_CTRL_IN (LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_IN)
    #define UHFMAN_CH340_CTRL_OUT (LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_OUT)
    #define UHFMAN_CH340_VENDOR_ID 0x1a86
    #define UHFMAN_CH340_PRODUCT_ID 0x7523
    #define UHFMAN_CH340_USB_IFACE_IX 0
    #if UHFMAN_CH340_USE_KERNEL_DRIVER == 1
        #include <fcntl.h> 

        #ifndef __USE_MISC
        #define __USE_MISC
        #endif // __USE_MISC (this is to make sure cfmakeraw is defined)
        //#define __GNU_SOURCE
        #include <asm-generic/termbits-common.h>
        #include <termios.h>
    #endif
#else
    #error "Unknown UHFMAN_DEVICE_CONNECTION_TYPE"
#endif

#if UHFMAN_DEVICE_CONNECTION_TYPE == UHFMAN_DEVICE_CONNECTION_TYPE_CH340_USB && UHFMAN_CH340_USE_KERNEL_DRIVER == 0
#define UHFMAN_DETACH_KERNEL_DRIVERS_ERR_SUCCESS 0
#define UHFMAN_DETACH_KERNEL_DRIVERS_ERR_NO_DEVICE_WHEN_CHECKING 1
#define UHFMAN_DETACH_KERNEL_DRIVERS_ERR_NOT_SUPPORTED_WHEN_CHECKING 2
#define UHFMAN_DETACH_KERNEL_DRIVERS_ERR_INVALID_PARAM 3
#define UHFMAN_DETACH_KERNEL_DRIVERS_ERR_NO_DEVICE_WHEN_DETACHING 4
#define UHFMAN_DETACH_KERNEL_DRIVERS_ERR_NOT_SUPPORTED_WHEN_DETACHING 5
#define UHFMAN_DETACH_KERNEL_DRIVERS_ERR_OTHER_WHEN_CHECKING 6
#define UHFMAN_DETACH_KERNEL_DRIVERS_ERR_OTHER_WHEN_DETACHING 7
static int uhfman_detach_kernel_drivers(uhfman_ctx_t* pCtx) {
    int rv = libusb_kernel_driver_active(pCtx->handle, UHFMAN_CH340_USB_IFACE_IX);

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
            rv = libusb_detach_kernel_driver(pCtx->handle, UHFMAN_CH340_USB_IFACE_IX);
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
#endif // UHFMAN_DEVICE_CONNECTION_TYPE == UHFMAN_DEVICE_CONNECTION_TYPE_CH340_USB AND UHFMAN_CH340_USE_KERNEL_DRIVER == 0

#if (UHFMAN_DEVICE_CONNECTION_TYPE == UHFMAN_DEVICE_CONNECTION_TYPE_CH340_USB) && (UHFMAN_CH340_USE_KERNEL_DRIVER == 1)
static int uhfman_usbserial_set_interface_attribs (int fd, int speed, int parity)
{
        struct termios tty;
        if (tcgetattr (fd, &tty) != 0)
        {
                fprintf (stderr, "error %d from tcgetattr", errno);
                return -1;
        }

        cfsetospeed (&tty, speed);
        cfsetispeed (&tty, speed);

        tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;     // 8-bit chars
        // disable IGNBRK for mismatched speed tests; otherwise receive break
        // as \000 chars
        tty.c_iflag &= ~IGNBRK;         // disable break processing
        tty.c_lflag = 0;                // no signaling chars, no echo,
                                        // no canonical processing
        tty.c_oflag = 0;                // no remapping, no delays

        tty.c_cc[VMIN]  = 0;            // read doesn't block
        tty.c_cc[VTIME] = 5;            // 0.5 seconds read timeout

        tty.c_iflag &= ~(IXON | IXOFF | IXANY); // shut off xon/xoff ctrl

        tty.c_cflag |= (CLOCAL | CREAD);// ignore modem controls,
                                        // enable reading
        tty.c_cflag &= ~(PARENB | PARODD);      // shut off parity
        tty.c_cflag |= parity;
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CRTSCTS;

        if (tcsetattr (fd, TCSANOW, &tty) != 0)
        {
                fprintf (stderr, "error %d from tcsetattr", errno);
                return -1;
        }
        return 0;
}

static void uhfman_usbserial_set_blocking (int fd, int should_block)
{
        struct termios tty;
        memset (&tty, 0, sizeof tty);
        if (tcgetattr (fd, &tty) != 0)
        {
                fprintf (stderr, "error %d from tggetattr", errno);
                return;
        }

        tty.c_cc[VMIN]  = should_block ? 1 : 0;
        tty.c_cc[VTIME] = 5;            // 0.5 seconds read timeout

        if (tcsetattr (fd, TCSANOW, &tty) != 0)
                fprintf (stderr, "error %d setting term attributes", errno);
}
#endif // UHFMAN_DEVICE_CONNECTION_TYPE == UHFMAN_DEVICE_CONNECTION_TYPE_CH340_USB AND UHFMAN_CH340_USE_KERNEL_DRIVER == 1


static void uhfman_debug_print_bits(void const * const ptr, size_t const size)
{
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;
    
    for (i = size-1; i >= 0; i--) {
        for (j = 7; j >= 0; j--) {
            byte = (b[i] >> j) & 1;
            printf("%u", byte);
        }
    }
}

uhfman_err_t uhfman_device_take(uhfman_ctx_t *pCtx_out) {
#if UHFMAN_DEVICE_CONNECTION_TYPE == UHFMAN_DEVICE_CONNECTION_TYPE_UART
#error "UHFMAN_DEVICE_CONNECTION_TYPE_UART not supported for now"
#elif UHFMAN_DEVICE_CONNECTION_TYPE == UHFMAN_DEVICE_CONNECTION_TYPE_CH340_USB
    #if UHFMAN_CH340_USE_KERNEL_DRIVER == 0
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
    handle = libusb_open_device_with_vid_pid(context, UHFMAN_CH340_VENDOR_ID, UHFMAN_CH340_PRODUCT_ID);
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
    r = libusb_claim_interface(handle, UHFMAN_CH340_USB_IFACE_IX);
    if (r < 0) {
        fprintf(stderr, "Error claiming interface: %s\n", libusb_error_name(r));
        #if UHFMAN_USE_DEBUG_EXTENSIONS == 1
        uhfman_debug_errno();
        #endif
        libusb_close(handle);
        libusb_exit(context);
        return UHFMAN_TAKE_ERR_INTERFACE_CLAIM;
    }

    // Initialize ch340
    r = ch340_init(handle);
    if (r < 0) {
        fprintf(stderr, "Error initializing ch340: %d\n", r);
        libusb_release_interface(handle, UHFMAN_CH340_USB_IFACE_IX);
        libusb_close(handle);
        libusb_exit(context);
        return UHFMAN_TAKE_ERR_BRIDGE_INIT_FAIL;
    }

    return UHFMAN_TAKE_ERR_SUCCESS;
    #elif UHFMAN_CH340_USE_KERNEL_DRIVER
        #ifndef UHFMAN_CH340_PORT_NAME
            #error "UHFMAN_CH340_PORT_NAME is undefined"
        #endif
        int fd = open(UHFMAN_CH340_PORT_NAME, O_RDWR | O_NOCTTY | O_SYNC | O_EXCL);
        if (fd < 0) {
            fprintf(stderr, "Error %d opening %s: %s", errno, UHFMAN_CH340_PORT_NAME, strerror (errno));
            return -1;
        }

        //uhfman_usbserial_set_interface_attribs(fd, B9600, 0); // set speed to 9600 bps, 8n1 (no parity)
        //uhfman_usbserial_set_blocking(fd, 0); // set no blocking

        if (!isatty(fd)) {
            fprintf(stderr, "%s is not a tty\n", UHFMAN_CH340_PORT_NAME);
            close(fd);
            return -1;
        } else {
            fprintf(stdout, "%s is a tty\n", UHFMAN_CH340_PORT_NAME);
        }

        struct termios config;
        if (tcgetattr(fd, &config) < 0) {
            fprintf(stderr, "Error getting termios attributes: %s\n", strerror(errno));
            close(fd);
            return -1;
        }

        cfmakeraw(&config);
        /*
            Calling cfmakeraw should be equivalent to using 
            t->c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP |
			INLCR | IGNCR | ICRNL | IXON);
            t->c_oflag &= ~OPOST;
            t->c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
            t->c_cflag &= ~(CSIZE | PARENB);
            t->c_cflag |= CS8;
            t->c_cc[VMIN] = 1;
            t->c_cc[VTIME] = 0;

            Compare with https://github.com/aligrudi/neatlibc/blob/master/termios.c
        */

        cfsetispeed(&config, B115200);
        cfsetospeed(&config, B115200);

        // config.c_iflag = IGNBRK | IGNPAR | IGNCR;
        // config.c_iflag &= ~(BRKINT | PARMRK | INPCK | ISTRIP | INLCR | ICRNL | IXANY);

        // //config.c_oflag &= ~(OPOST | ONLCR);
        // config.c_oflag = 0;

        // // TODO: Check for termios config issues; add 0x11 command for ypd-r200

        // config.c_lflag &= ~(ICANON | ECHO); // ~ICANON for non-canonical mode, ~ECHO for no echo
        // config.c_cc[VMIN] = 1; // Read at least 1 byte
        // config.c_cc[VTIME] = 0; // No timeout
        config.c_cflag |= CRTSCTS;

        ///<Print all termios attributes in binary for debugging>
        fprintf(stdout, "Termios attributes (binary):\n");
        fprintf(stdout, "c_iflag: "); uhfman_debug_print_bits(&config.c_iflag, sizeof(config.c_iflag)); fprintf(stdout, "\n");
        fprintf(stdout, "c_oflag: "); uhfman_debug_print_bits(&config.c_oflag, sizeof(config.c_oflag)); fprintf(stdout, "\n");
        fprintf(stdout, "c_cflag: "); uhfman_debug_print_bits(&config.c_cflag, sizeof(config.c_cflag)); fprintf(stdout, "\n");
        fprintf(stdout, "c_lflag: "); uhfman_debug_print_bits(&config.c_lflag, sizeof(config.c_lflag)); fprintf(stdout, "\n");
        fprintf(stdout, "c_line: "); uhfman_debug_print_bits(&config.c_line, sizeof(config.c_line)); fprintf(stdout, "\n");
        fprintf(stdout, "c_cc: ");
        for (int i = 0; i < NCCS; i++) {
            fprintf(stdout, "c_cc[%d]: ", i); uhfman_debug_print_bits(&config.c_cc[i], sizeof(config.c_cc[i])); fprintf(stdout, "\n");
        }
        fprintf(stdout, "c_ispeed: "); uhfman_debug_print_bits(&config.c_ispeed, sizeof(config.c_ispeed)); fprintf(stdout, "\n");
        fprintf(stdout, "c_ospeed: "); uhfman_debug_print_bits(&config.c_ospeed, sizeof(config.c_ospeed)); fprintf(stdout, "\n");
        ///</Print all termios attributes for debugging>

        if (tcsetattr(fd, TCSANOW, &config) < 0) {
            fprintf(stderr, "Error setting termios attributes: %s\n", strerror(errno));
            close(fd);
            return -1;
        } else {
            fprintf(stdout, "Termios attributes set successfully\n");
        }

        tcflush(fd, TCIOFLUSH); // Flush just in case there is any garbage in the buffer

        pCtx_out->fd = fd;
        //printf("EXITING FOR NOW\n"); exit(EXIT_SUCCESS);
        // int rv = ypdr200_x11(pCtx_out, YPDR200_X11_PARAM_BAUD_RATE_115200); // Set baud rate to 115200 bps
        // if (rv != YPDR200_X11_ERR_SUCCESS) {
        //     fprintf(stderr, "Error setting baud rate: %d\n", rv);
        //     close(fd);
        //     return -1; // TODO replace those -1 codes with proper error codes
        // } else {
        //     fprintf(stdout, "Baud rate set successfully\n");
        // }
        return UHFMAN_TAKE_ERR_SUCCESS;
    #else
    #error "Invalid value of UHFMAN_CH340_USE_KERNEL_DRIVER"pCtx_out
    #endif
#else
#error "Unknown UHFMAN_DEVICE_CONNECTION_TYPE"
#endif
}

void uhfman_device_release(uhfman_ctx_t *pCtx) {
#if UHFMAN_DEVICE_CONNECTION_TYPE == UHFMAN_DEVICE_CONNECTION_TYPE_UART
#error "UHFMAN_DEVICE_CONNECTION_TYPE_UART not supported for now"
#elif UHFMAN_DEVICE_CONNECTION_TYPE == UHFMAN_DEVICE_CONNECTION_TYPE_CH340_USB
    #if UHFMAN_CH340_USE_KERNEL_DRIVER == 0
    if (0 != libusb_release_interface(pCtx->handle, 0)) {
        fprintf(stderr, "Error releasing interface\n");
    }
    libusb_close(pCtx->handle);
    libusb_exit(pCtx->context);
    #else
    close(pCtx->fd);
    #endif
#else
#error "Unknown UHFMAN_DEVICE_CONNECTION_TYPE"
#endif
}

uhfman_err_t uhfman_get_hardware_version(uhfman_ctx_t* pCtx, char** ppcVersion_out) {
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    ypdr200_resp_err_code_t rerr = 0;
    int rv = ypdr200_x03(pCtx, YPDR200_X03_PARAM_HARDWARE_VERSION, ppcVersion_out, &rerr);
    switch (rv) {
        case YPDR200_X03_ERR_SUCCESS:
            return UHFMAN_GET_HARDWARE_VERSION_ERR_SUCCESS;
        case YPDR200_X03_ERR_SEND_COMMAND:
            return UHFMAN_GET_HARDWARE_VERSION_ERR_SEND_COMMAND;
        case YPDR200_X03_ERR_READ_RESPONSE:
            return UHFMAN_GET_HARDWARE_VERSION_ERR_READ_RESPONSE;
        case YPDR200_X03_ERR_ERROR_RESPONSE:
            fprintf(stderr, "** Response frame was an error frame containing error code 0x%02X **\n", (uint8_t)rerr);
            return UHFMAN_GET_HARDWARE_VERSION_ERR_ERROR_RESPONSE;
        default:
            fprintf(stderr, "Unknown error from ypdr200_x03: %d\n", rv);
            return UHFMAN_GET_HARDWARE_VERSION_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_GET_HARDWARE_VERSION_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}

uhfman_err_t uhfman_get_software_version(uhfman_ctx_t* pCtx, char** ppcVersion_out) {
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    ypdr200_resp_err_code_t rerr = 0;
    int rv = ypdr200_x03(pCtx, YPDR200_X03_PARAM_SOFTWARE_VERSION, ppcVersion_out, &rerr);
    switch (rv) {
        case YPDR200_X03_ERR_SUCCESS:
            return UHFMAN_GET_SOFTWARE_VERSION_ERR_SUCCESS;
        case YPDR200_X03_ERR_SEND_COMMAND:
            return UHFMAN_GET_SOFTWARE_VERSION_ERR_SEND_COMMAND;
        case YPDR200_X03_ERR_READ_RESPONSE:
            return UHFMAN_GET_SOFTWARE_VERSION_ERR_READ_RESPONSE;
        case YPDR200_X03_ERR_ERROR_RESPONSE:
            fprintf(stderr, "** Response frame was an error frame containing error code 0x%02X **\n", (uint8_t)rerr);
            return UHFMAN_GET_SOFTWARE_VERSION_ERR_ERROR_RESPONSE;
        default:
            fprintf(stderr, "Unknown error from ypdr200_x03: %d\n", rv);
            return UHFMAN_GET_SOFTWARE_VERSION_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_GET_SOFTWARE_VERSION_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}

uhfman_err_t uhfman_get_manufacturer(uhfman_ctx_t* pCtx, char** ppcManufacturer_out) {
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    ypdr200_resp_err_code_t rerr = 0;
    int rv = ypdr200_x03(pCtx, YPDR200_X03_PARAM_MANUFACTURER, ppcManufacturer_out, &rerr);
    switch (rv) {
        case YPDR200_X03_ERR_SUCCESS:
            return UHFMAN_GET_MANUFACTURER_ERR_SUCCESS;
        case YPDR200_X03_ERR_SEND_COMMAND:
            return UHFMAN_GET_MANUFACTURER_ERR_SEND_COMMAND;
        case YPDR200_X03_ERR_READ_RESPONSE:
            return UHFMAN_GET_MANUFACTURER_ERR_READ_RESPONSE;
        case YPDR200_X03_ERR_ERROR_RESPONSE:
            fprintf(stderr, "** Response frame was an error frame containing error code 0x%02X **\n", (uint8_t)rerr);
            return UHFMAN_GET_MANUFACTURER_VERSION_ERR_ERROR_RESPONSE;
        default:
            fprintf(stderr, "Unknown error from ypdr200_x03: %d\n", rv);
            return UHFMAN_GET_MANUFACTURER_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_GET_MANUFACTURER_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}

uhfman_err_t uhfman_dbg_get_select_param(uhfman_ctx_t* pCtx) {
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    ypdr200_resp_err_code_t rerr = 0;
    ypdr200_x0b_resp_param_t respParam;
    int rv = ypdr200_x0b(pCtx, &respParam, &rerr);
    switch (rv) {
        case YPDR200_X0B_ERR_SUCCESS:
            fprintf(stdout, "Select parameter: target=0x%02X, action=0x%02X, memBank=0x%02X, ptr=0x%02X%02X%02X%02X, maskLen=%d, truncate=%d Mask=[", respParam.hdr.target, respParam.hdr.action, respParam.hdr.memBank, respParam.hdr.ptr[0], respParam.hdr.ptr[1], respParam.hdr.ptr[2], respParam.hdr.ptr[3], respParam.hdr.maskLen, respParam.hdr.truncate);
            for (int i = 0; i < respParam.hdr.maskLen; i++) {
                fprintf(stdout, "%02X", respParam.pMask[i]);
            }
            fprintf(stdout, "]\n");
            ypdr200_x0b_resp_param_dispose(&respParam);
            return UHFMAN_GET_SELECT_PARAM_ERR_SUCCESS;
        case YPDR200_X0B_ERR_SEND_COMMAND:
            return UHFMAN_GET_SELECT_PARAM_ERR_SEND_COMMAND;
        case YPDR200_X0B_ERR_READ_RESPONSE:
            return UHFMAN_GET_SELECT_PARAM_ERR_READ_RESPONSE;
        case YPDR200_X0B_ERR_ERROR_RESPONSE:
            fprintf(stderr, "** Response frame was an error frame containing error code 0x%02X **\n", (uint8_t)rerr);
            return UHFMAN_GET_SELECT_PARAM_ERR_ERROR_RESPONSE;
        default:
            fprintf(stderr, "Unknown error from ypdr200_x0b: %d\n", rv);
            return UHFMAN_GET_SELECT_PARAM_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_GET_SELECT_PARAM_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}

uhfman_err_t uhfman_dbg_get_query_params(uhfman_ctx_t* pCtx) {
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    ypdr200_resp_err_code_t rerr = 0;
    ypdr200_x0d_resp_param_t respParam;
    int rv = ypdr200_x0d(pCtx, &respParam, &rerr);
    switch (rv) {
        case YPDR200_X0D_ERR_SUCCESS:
            fprintf(stdout, "Query parameters: dr=0x%02X, m=0x%02X, trext=0x%02X, sel=0x%02X, session=0x%02X, target=0x%02X q=0x%02X\n", respParam.dr, respParam.m, respParam.trext, respParam.sel, respParam.session, respParam.target, respParam.q);
            return UHFMAN_GET_QUERY_PARAMS_ERR_SUCCESS;
        case YPDR200_X0D_ERR_SEND_COMMAND:
            return UHFMAN_GET_QUERY_PARAMS_ERR_SEND_COMMAND;
        case YPDR200_X0D_ERR_READ_RESPONSE:
            return UHFMAN_GET_QUERY_PARAMS_ERR_READ_RESPONSE;
        case YPDR200_X0D_ERR_ERROR_RESPONSE:
            fprintf(stderr, "** Response frame was an error frame containing error code 0x%02X **\n", (uint8_t)rerr);
            return UHFMAN_GET_QUERY_PARAMS_ERR_ERROR_RESPONSE;
        default:
            fprintf(stderr, "Unknown error from ypdr200_x0d: %d\n", rv);
            return UHFMAN_GET_QUERY_PARAMS_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_GET_QUERY_PARAMS_ERR_UNKNOWN_DEVICE_MODEL; // TODO Change to #error or remove, because generating the error once is enough (applies to other functions here as well)
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}

uhfman_err_t uhfman_dbg_get_working_channel(uhfman_ctx_t* pCtx) {
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    ypdr200_resp_err_code_t rerr = 0;
    uint8_t chIndex = 0;
    int rv = ypdr200_xaa(pCtx, &chIndex, &rerr);
    switch (rv) {
        case YPDR200_XAA_ERR_SUCCESS:
            fprintf(stdout, "Working channel: 0x%02X\n", chIndex);
            return UHFMAN_GET_WORKING_CHANNEL_ERR_SUCCESS;
        case YPDR200_XAA_ERR_SEND_COMMAND:
            return UHFMAN_GET_WORKING_CHANNEL_ERR_SEND_COMMAND;
        case YPDR200_XAA_ERR_READ_RESPONSE:
            return UHFMAN_GET_WORKING_CHANNEL_ERR_READ_RESPONSE;
        case YPDR200_XAA_ERR_ERROR_RESPONSE:
            fprintf(stderr, "** Response frame was an error frame containing error code 0x%02X **\n", (uint8_t)rerr);
            return UHFMAN_GET_WORKING_CHANNEL_ERR_ERROR_RESPONSE;
        default:
            fprintf(stderr, "Unknown error from ypdr200_xaa: %d\n", rv);
            return UHFMAN_GET_WORKING_CHANNEL_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_GET_WORKING_CHANNEL_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}

uhfman_err_t uhfman_dbg_get_work_area(uhfman_ctx_t* pCtx) {
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    ypdr200_resp_err_code_t rerr = 0;
    ypdr200_x08_region_t region;
    int rv = ypdr200_x08(pCtx, &region, &rerr);
    switch (rv) {
        case YPDR200_X08_ERR_SUCCESS:
            fprintf(stdout, "Work area/Region: 0x%02X\n", region);
            return UHFMAN_GET_WORK_AREA_ERR_SUCCESS;
        case YPDR200_X08_ERR_SEND_COMMAND:
            return UHFMAN_GET_WORK_AREA_ERR_SEND_COMMAND;
        case YPDR200_X08_ERR_READ_RESPONSE:
            return UHFMAN_GET_WORK_AREA_ERR_READ_RESPONSE;
        case YPDR200_X08_ERR_ERROR_RESPONSE:
            fprintf(stderr, "** Response frame was an error frame containing error code 0x%02X **\n", (uint8_t)rerr);
            return UHFMAN_GET_WORK_AREA_ERR_ERROR_RESPONSE;
        default:
            fprintf(stderr, "Unknown error from ypdr200_x08: %d\n", rv);
            return UHFMAN_GET_WORK_AREA_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_GET_WORK_AREA_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}

uhfman_err_t uhfman_dbg_get_transmit_power(uhfman_ctx_t* pCtx) {
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    ypdr200_resp_err_code_t rerr = 0;
    uint16_t powerLevel = 0;
    int rv = ypdr200_xb7(pCtx, &powerLevel, &rerr);
    switch (rv) {
        case YPDR200_XB7_ERR_SUCCESS:
            fprintf(stdout, "Transmit power level: 0x%02X\n", powerLevel);
            return UHFMAN_GET_TRANSMIT_POWER_ERR_SUCCESS;
        case YPDR200_XB7_ERR_SEND_COMMAND:
            return UHFMAN_GET_TRANSMIT_POWER_ERR_SEND_COMMAND;
        case YPDR200_XB7_ERR_READ_RESPONSE:
            return UHFMAN_GET_TRANSMIT_POWER_ERR_READ_RESPONSE;
        case YPDR200_XB7_ERR_ERROR_RESPONSE:
            fprintf(stderr, "** Response frame was an error frame containing error code 0x%02X **\n", (uint8_t)rerr);
            return UHFMAN_GET_TRANSMIT_POWER_ERR_ERROR_RESPONSE;
        default:
            fprintf(stderr, "Unknown error from ypdr200_xb7: %d\n", rv);
            return UHFMAN_GET_TRANSMIT_POWER_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_GET_TRANSMIT_POWER_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}

uhfman_err_t uhfman_dbg_get_demod_params(uhfman_ctx_t* pCtx) {
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    ypdr200_resp_err_code_t rerr = 0;
    ypdr200_xf1_rx_demod_params_t respParam;
    int rv = ypdr200_xf1(pCtx, &respParam, &rerr);
    switch (rv) {
        case YPDR200_XF1_ERR_SUCCESS:
            uint16_t thrd = ((uint16_t)(respParam.thrdMsb) << 8) | (uint16_t)(respParam.thrdLsb);
            fprintf(stdout, "Demodulator parameters: mixer_G: 0x%02X, if_G: 0x%02X, thrdMsb: 0x%02X, thrdLsb: 0x%02X (thrd: 0x%04X)\n", respParam.mixer_G, respParam.if_G, respParam.thrdMsb, respParam.thrdLsb, thrd);
            return UHFMAN_GET_DEMOD_PARAMS_ERR_SUCCESS;
        case YPDR200_XF1_ERR_SEND_COMMAND:
            return UHFMAN_GET_DEMOD_PARAMS_ERR_SEND_COMMAND;
        case YPDR200_XF1_ERR_READ_RESPONSE:
            return UHFMAN_GET_DEMOD_PARAMS_ERR_READ_RESPONSE;
        case YPDR200_XF1_ERR_ERROR_RESPONSE:
            fprintf(stderr, "** Response frame was an error frame containing error code 0x%02X **\n", (uint8_t)rerr);
            return UHFMAN_GET_DEMOD_PARAMS_ERR_ERROR_RESPONSE;
        default:
            fprintf(stderr, "Unknown error from ypdr200_xf1: %d\n", rv);
            return UHFMAN_GET_DEMOD_PARAMS_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_GET_DEMOD_PARAMS_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}

#if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
static void uhfman_dbg_single_polling_notification_handler(ypdr200_x22_ntf_param_t ntfParam, const void* pUserData) {
    if ((ntfParam.crc[0] == 0x07 && ntfParam.crc[1] == 0x75)
        || (ntfParam.crc[0] == 0xDC && ntfParam.crc[1] == 0x20)) {
        return; // Ignore those tags
    }
    fprintf(stdout, "Polling notification: rssi=0x%02X, pc=0x%04X, epc=[ ", ntfParam.rssi, ((uint16_t)(ntfParam.pc[0]) << 8) | (uint16_t)(ntfParam.pc[1]));
    for (int i = 0; i < YPDR200_X22_NTF_PARAM_EPC_LENGTH; i++) { //TODO variabilize (related to #ae3759b4)
        fprintf(stdout, "%02X ", ntfParam.epc[i]);
    }
    fprintf(stdout, "], crc=0x%04X\n", ((uint16_t)(ntfParam.crc[0]) << 8) | (uint16_t)(ntfParam.crc[1]));
}
#else
#error "Unknown UHFMAN_DEVICE_MODEL"
#endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200

uhfman_err_t uhfman_dbg_single_polling(uhfman_ctx_t* pCtx) {
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    ypdr200_resp_err_code_t rerr = 0;
    int rv = ypdr200_x22(pCtx, &rerr, uhfman_dbg_single_polling_notification_handler, NULL);
    switch (rv) {
        case YPDR200_X22_ERR_SUCCESS:
            return UHFMAN_SINGLE_POLLING_ERR_SUCCESS;
        case YPDR200_X22_ERR_SEND_COMMAND:
            return UHFMAN_SINGLE_POLLING_ERR_SEND_COMMAND;
        case YPDR200_X22_ERR_READ_RESPONSE:
            return UHFMAN_SINGLE_POLLING_ERR_READ_RESPONSE;
        case YPDR200_X22_ERR_ERROR_RESPONSE:
            fprintf(stderr, "** Response frame was an error frame containing error code 0x%02X **\n", (uint8_t)rerr);
            return UHFMAN_SINGLE_POLLING_ERR_ERROR_RESPONSE;
        case YPDR200_X22_ERR_READ_NOTIFICATION:
            return UHFMAN_SINGLE_POLLING_ERR_READ_NOTIFICATION;
        case YPDR200_X22_ERR_UNEXPECTED_FRAME_TYPE:
            return UHFMAN_SINGLE_POLLING_ERR_UNEXPECTED_FRAME_TYPE;
        default:
            fprintf(stderr, "Unknown error from ypdr200_x22: %d\n", rv);
            return UHFMAN_SINGLE_POLLING_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_DBG_SINGLE_POLLING_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}

uhfman_err_t uhfman_dbg_multiple_polling(uhfman_ctx_t* pCtx) {
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    ypdr200_x27_req_param_t param = ypdr200_x27_req_param_make(0xffff);
    ypdr200_resp_err_code_t rerr = 0;
    int rv = ypdr200_x27(pCtx, param, &rerr, uhfman_dbg_single_polling_notification_handler, NULL);
    switch (rv) {
        case YPDR200_X27_ERR_SUCCESS:
            return UHFMAN_SINGLE_POLLING_ERR_SUCCESS;
        case YPDR200_X27_ERR_SEND_COMMAND:
            return UHFMAN_SINGLE_POLLING_ERR_SEND_COMMAND;
        case YPDR200_X27_ERR_READ_RESPONSE:
            return UHFMAN_SINGLE_POLLING_ERR_READ_RESPONSE;
        case YPDR200_X27_ERR_ERROR_RESPONSE:
            fprintf(stderr, "** Response frame was an error frame containing error code 0x%02X **\n", (uint8_t)rerr);
            return UHFMAN_SINGLE_POLLING_ERR_ERROR_RESPONSE;
        case YPDR200_X27_ERR_READ_NOTIFICATION:
            return UHFMAN_SINGLE_POLLING_ERR_READ_NOTIFICATION;
        case YPDR200_X27_ERR_UNEXPECTED_FRAME_TYPE:
            return UHFMAN_SINGLE_POLLING_ERR_UNEXPECTED_FRAME_TYPE;
        default:
            fprintf(stderr, "Unknown error from ypdr200_x27: %d\n", rv);
            return UHFMAN_SINGLE_POLLING_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_DBG_SINGLE_POLLING_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}