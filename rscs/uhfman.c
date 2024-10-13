#include "uhfman.h"
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>

#include "log.h"
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
                    LOG_I("Kernel driver detached successfully");
                    return UHFMAN_DETACH_KERNEL_DRIVERS_ERR_SUCCESS;
                case LIBUSB_ERROR_INVALID_PARAM:
                    return UHFMAN_DETACH_KERNEL_DRIVERS_ERR_INVALID_PARAM;
                case LIBUSB_ERROR_NO_DEVICE:
                    return UHFMAN_DETACH_KERNEL_DRIVERS_ERR_NO_DEVICE_WHEN_DETACHING;
                case LIBUSB_ERROR_NOT_SUPPORTED:
                    return UHFMAN_DETACH_KERNEL_DRIVERS_ERR_NOT_SUPPORTED_WHEN_DETACHING;
                default:
                    LOG_E("Unknown error when detaching kernel driver: %d", rv);
                    return UHFMAN_DETACH_KERNEL_DRIVERS_ERR_OTHER_WHEN_DETACHING;
            }
        default:
            LOG_E("Unknown error when checking kernel driver status: %d", rv);
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
                LOG_E("error %d from tcgetattr", errno);
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
                LOG_E("error %d from tcsetattr", errno);
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
                LOG_E("error %d from tggetattr", errno);
                return;
        }

        tty.c_cc[VMIN]  = should_block ? 1 : 0;
        tty.c_cc[VTIME] = 5;            // 0.5 seconds read timeout

        if (tcsetattr (fd, TCSANOW, &tty) != 0)
                LOG_E("error %d setting term attributes", errno);
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
            LOG_D_CTBC("%u", byte);
        }
    }
}

static void uhfman_ctx_config_init(uhfman_ctx_t* pCtx) {
    pCtx->_config.select_params.target = UHFMAN_SELECT_TARGET_UNKNOWN;
    pCtx->_config.select_params.action = UHFMAN_SELECT_ACTION_UNKNOWN;
    pCtx->_config.select_params.memBank = UHFMAN_SELECT_MEMBANK_UNKNOWN;
    pCtx->_config.select_params.ptr = 0;
    pCtx->_config.select_params.maskLen = 0;
    pCtx->_config.select_params.truncate = 0;
    pCtx->_config.select_params.pMask = NULL;
    pCtx->_config.select_mode = UHFMAN_SELECT_MODE_UNKNOWN;
    pCtx->_config.query_params.sel = UHFMAN_QUERY_SEL_UNKNOWN;
    pCtx->_config.query_params.session = UHFMAN_QUERY_SESSION_UNKNOWN;
    pCtx->_config.query_params.target = UHFMAN_QUERY_TARGET_UNKNOWN;
    pCtx->_config.query_params.q = 0;
    pCtx->_config.txPower = NAN;
    pCtx->_config.flags = 0; // no params initialized with hardware
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
        LOG_E("Error initializing libusb: %s", libusb_error_name(r));
        return UHFMAN_TAKE_ERR_LIBUSB_INIT;
    }

    // Open the device 
    handle = libusb_open_device_with_vid_pid(context, UHFMAN_CH340_VENDOR_ID, UHFMAN_CH340_PRODUCT_ID);
    //libusb_open(experiment_global_dev_ptr, &handle);

    if (handle == NULL) {
        if (errno == 0) {
            LOG_E("Device found, but handle is NULL. Is it a driver issue?"); // TODO Return something different than UHFMAN_TAKE_ERR_DEVICE_NOT_FOUND
        }
        else {
            LOG_E("Error finding USB device\n");
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
    LOG_D("Detaching kernel drivers if needed...");
    r = uhfman_detach_kernel_drivers(pCtx_out);
    if (r != UHFMAN_DETACH_KERNEL_DRIVERS_ERR_SUCCESS) {
        LOG_E("Error detaching kernel drivers: %d", r);
        libusb_close(handle);
        libusb_exit(context);
        return UHFMAN_TAKE_ERR_INTERFACE_CLAIM;
    }

    // Claim the interface (assuming interface UHFMAN_USB_IFACE_IX)
    r = libusb_claim_interface(handle, UHFMAN_CH340_USB_IFACE_IX);
    if (r < 0) {
        LOG_E("Error claiming interface: %s", libusb_error_name(r));
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
        LOG_E("Error initializing ch340: %d", r);
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
            LOG_E("Error %d opening %s: %s", errno, UHFMAN_CH340_PORT_NAME, strerror (errno));
            return -1;
        }

        //uhfman_usbserial_set_interface_attribs(fd, B9600, 0); // set speed to 9600 bps, 8n1 (no parity)
        //uhfman_usbserial_set_blocking(fd, 0); // set no blocking

        if (!isatty(fd)) {
            LOG_E("%s is not a tty", UHFMAN_CH340_PORT_NAME);
            close(fd);
            return -1;
        } else {
            LOG_D("%s is a tty", UHFMAN_CH340_PORT_NAME);
        }

        struct termios config;
        if (tcgetattr(fd, &config) < 0) {
            LOG_E("Error getting termios attributes: %s", strerror(errno));
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
        LOG_V("Termios attributes (binary): ");
        LOG_V_TBC("c_iflag: "); uhfman_debug_print_bits(&config.c_iflag, sizeof(config.c_iflag)); LOG_V_CFIN("");
        LOG_V_TBC("c_oflag: "); uhfman_debug_print_bits(&config.c_oflag, sizeof(config.c_oflag)); LOG_V_CFIN("");
        LOG_V_TBC("c_cflag: "); uhfman_debug_print_bits(&config.c_cflag, sizeof(config.c_cflag)); LOG_V_CFIN("");
        LOG_V_TBC("c_lflag: "); uhfman_debug_print_bits(&config.c_lflag, sizeof(config.c_lflag)); LOG_V_CFIN("");
        LOG_V_TBC("c_line: "); uhfman_debug_print_bits(&config.c_line, sizeof(config.c_line)); LOG_V_CFIN("");
        LOG_V("c_cc: ");
        for (int i = 0; i < NCCS; i++) {
            LOG_V_TBC("c_cc[%d]: ", i); uhfman_debug_print_bits(&config.c_cc[i], sizeof(config.c_cc[i])); LOG_V_CFIN("");
        }
        LOG_V_TBC("c_ispeed: "); uhfman_debug_print_bits(&config.c_ispeed, sizeof(config.c_ispeed)); LOG_V_CFIN("");
        LOG_V_TBC("c_ospeed: "); uhfman_debug_print_bits(&config.c_ospeed, sizeof(config.c_ospeed)); LOG_V_CFIN("");
        ///</Print all termios attributes for debugging>

        if (tcsetattr(fd, TCSANOW, &config) < 0) {
            LOG_E("Error setting termios attributes: %s", strerror(errno));
            close(fd);
            return -1;
        } else {
            LOG_D("Termios attributes set successfully");
        }

        tcflush(fd, TCIOFLUSH); // Flush just in case there is any garbage in the buffers

        pCtx_out->fd = fd;
        uhfman_ctx_config_init(pCtx_out);
        
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
        LOG_E("Error releasing interface");
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
            LOG_W("** Response frame was an error frame containing error code 0x%02X **", (uint8_t)rerr);
            return UHFMAN_GET_HARDWARE_VERSION_ERR_ERROR_RESPONSE;
        default:
            LOG_W("Unknown error from ypdr200_x03: %d", rv);
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
            LOG_W("** Response frame was an error frame containing error code 0x%02X **", (uint8_t)rerr);
            return UHFMAN_GET_SOFTWARE_VERSION_ERR_ERROR_RESPONSE;
        default:
            LOG_W("Unknown error from ypdr200_x03: %d", rv);
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
            LOG_W("** Response frame was an error frame containing error code 0x%02X **", (uint8_t)rerr);
            return UHFMAN_GET_MANUFACTURER_VERSION_ERR_ERROR_RESPONSE;
        default:
            fprintf(stderr, "Unknown error from ypdr200_x03: %d", rv);
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
            LOG_I_TBC("Select parameter: target=0x%02X, action=0x%02X, memBank=0x%02X, ptr=0x%02X%02X%02X%02X, maskLen=%d, truncate=%d Mask=[", respParam.hdr.target, respParam.hdr.action, respParam.hdr.memBank, respParam.hdr.ptr[0], respParam.hdr.ptr[1], respParam.hdr.ptr[2], respParam.hdr.ptr[3], respParam.hdr.maskLen, respParam.hdr.truncate);
            uint8_t mask_nbytes = (respParam.hdr.maskLen + 7) >> 3;
            for (int i = 0; i < mask_nbytes; i++) {
                LOG_I_CTBC("%02X ", respParam.pMask[i]);
            }
            LOG_I_CFIN("]");
            ypdr200_x0b_resp_param_dispose(&respParam);
            return UHFMAN_GET_SELECT_PARAM_ERR_SUCCESS;
        case YPDR200_X0B_ERR_SEND_COMMAND:
            return UHFMAN_GET_SELECT_PARAM_ERR_SEND_COMMAND;
        case YPDR200_X0B_ERR_READ_RESPONSE:
            return UHFMAN_GET_SELECT_PARAM_ERR_READ_RESPONSE;
        case YPDR200_X0B_ERR_ERROR_RESPONSE:
            LOG_W("** Response frame was an error frame containing error code 0x%02X **", (uint8_t)rerr);
            return UHFMAN_GET_SELECT_PARAM_ERR_ERROR_RESPONSE;
        default:
            LOG_W("Unknown error from ypdr200_x0b: %d", rv);
            return UHFMAN_GET_SELECT_PARAM_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_GET_SELECT_PARAM_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}

static void uhfman_config_select_params_update(uhfman_ctx_t* pCtx, 
                                                uint8_t target, 
                                                uint8_t action, 
                                                uint8_t memBank, 
                                                uint32_t ptr, 
                                                uint8_t maskLen, 
                                                uint8_t truncate, 
                                                const uint8_t* pMask) {
    pCtx->_config.select_params.target = target;
    pCtx->_config.select_params.action = action;
    pCtx->_config.select_params.memBank = memBank;
    pCtx->_config.select_params.ptr = ptr;
    pCtx->_config.select_params.maskLen = maskLen;
    pCtx->_config.select_params.truncate = truncate;
    pCtx->_config.select_params.pMask = (uint8_t*)malloc((maskLen + 7) >> 3);
    memcpy(pCtx->_config.select_params.pMask, pMask, (maskLen + 7) >> 3);
    pCtx->_config.flags |= UHFMAN_CTX_CONFIG_FLAG_SELECT_INITIALIZED;
}

static int uhfman_config_select_params_cache_cmp(uhfman_ctx_t* pCtx, 
                                                    uint8_t target, 
                                                    uint8_t action, 
                                                    uint8_t memBank, 
                                                    uint32_t ptr, 
                                                    uint8_t maskLen, 
                                                    uint8_t truncate, 
                                                    const uint8_t* pMask) {
    if ((pCtx->_config.flags & UHFMAN_CTX_CONFIG_FLAG_SELECT_INITIALIZED) == 0) {
        return 1;
    }
    if ((pCtx->_config.select_params.target != target)
    || (pCtx->_config.select_params.action != action)
    || (pCtx->_config.select_params.memBank != memBank)
    || (pCtx->_config.select_params.ptr != ptr)
    || (pCtx->_config.select_params.maskLen != maskLen)
    || (pCtx->_config.select_params.truncate != truncate)) {
        return 1;
    } else {
        if (!memcmp(pCtx->_config.select_params.pMask, pMask, maskLen >> 3)) {
            return 1;
        } else if (maskLen & 0x07) {
            if (pCtx->_config.select_params.pMask[maskLen >> 3] != pMask[maskLen >> 3]) {
                return 1;
            }
        }
    }
    return 0;
}

uhfman_err_t uhfman_set_select_param(uhfman_ctx_t* pCtx, 
                                     uint8_t target, 
                                     uint8_t action, 
                                     uint8_t memBank, 
                                     uint32_t ptr, 
                                     uint8_t maskLen, 
                                     uint8_t truncate, 
                                     const uint8_t* pMask) {
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    if (!uhfman_config_select_params_cache_cmp(pCtx, target, action, memBank, ptr, maskLen, truncate, pMask)) {
        return UHFMAN_SET_SELECT_PARAM_ERR_SUCCESS;
    }
    ypdr200_x0c_req_param_t reqParam = {
        .hdr = {
            .target = target,
            .action = action,
            .memBank = memBank,
            .ptr = {
                (ptr >> 24) & 0xFF,
                (ptr >> 16) & 0xFF,
                (ptr >> 8) & 0xFF,
                ptr & 0xFF
            },
            .maskLen = maskLen,
            .truncate = truncate
        },
        .pMask = (uint8_t*)pMask
    };
    ypdr200_resp_err_code_t rerr = 0;
    int rv = ypdr200_x0c(pCtx, &reqParam, &rerr);
    switch (rv) {
        case YPDR200_X0C_ERR_SUCCESS:
            uhfman_config_select_params_update(pCtx, target, action, memBank, ptr, maskLen, truncate, pMask);
            return UHFMAN_SET_SELECT_PARAM_ERR_SUCCESS;
        case YPDR200_X0C_ERR_SEND_COMMAND:
            return UHFMAN_SET_SELECT_PARAM_ERR_SEND_COMMAND;
        case YPDR200_X0C_ERR_READ_RESPONSE:
            return UHFMAN_SET_SELECT_PARAM_ERR_READ_RESPONSE;
        case YPDR200_X0C_ERR_ERROR_RESPONSE:
            LOG_W("** Response frame was an error frame containing error code 0x%02X **", (uint8_t)rerr);
            return UHFMAN_SET_SELECT_PARAM_ERR_ERROR_RESPONSE;
        default:
            LOG_W("Unknown error from ypdr200_x0c: %d", rv);
            return UHFMAN_SET_SELECT_PARAM_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_SET_SELECT_PARAM_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}

uint8_t uhfman_select_action(uint8_t uTagMatching, uint8_t uTagNotMatching) {
    /*
        a b | r      a b  | r
        0 1 | 000    0001 | 000
        0 3 | 001    0011 | 001
        3 1 | 010    1101 | 010
        2 3 | 011    1011 | 011
        1 0 | 100    0100 | 100
        1 3 | 101    0111 | 101
        3 0 | 110    1100 | 110
        3 2 | 111    1110 | 111
    */
    static const uint8_t uActionLUT[16] = {
        0xFF, 0x00, 0xFF, 0x01, 0x04, 0xFF, 0xFF, 0x05,
        0xFF, 0xFF, 0xFF, 0x03, 0x06, 0x02, 0x07, 0xFF
    };
    return uActionLUT[(uTagMatching << 2) | uTagNotMatching];
}

// uhfman_err_t uhfman_set_select_param_by_epc_code(uhfman_ctx_t* pCtx, const uint8_t* pEPC, size_t epcLen) {
//     uint8_t target = 
//     uhfman_err_t err = uhfman_set_select_param(pCtx,)
// }

static void uhfman_config_select_mode_update(uhfman_ctx_t* pCtx, 
                                                uhfman_select_mode_t mode) {
    pCtx->_config.select_mode = mode;
    pCtx->_config.flags |= UHFMAN_CTX_CONFIG_FLAG_SELECT_MODE_INITIALIZED;
}

static int uhfman_config_select_mode_cache_cmp(uhfman_ctx_t* pCtx, 
                                                    uhfman_select_mode_t mode) {
    if ((pCtx->_config.flags & UHFMAN_CTX_CONFIG_FLAG_SELECT_MODE_INITIALIZED) == 0) {
        return 1;
    }
    return (pCtx->_config.select_mode != mode);
}

uhfman_err_t uhfman_set_select_mode(uhfman_ctx_t* pCtx, uhfman_select_mode_t mode) {
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    if (!uhfman_config_select_mode_cache_cmp(pCtx, mode)) {
        return UHFMAN_SET_SELECT_MODE_ERR_SUCCESS;
    }
    uint8_t uMode = 0;
    switch (mode) {
        case UHFMAN_SELECT_MODE_ALWAYS:
            uMode = YPDR200_X12_MODE_ALWAYS;
            break;
        case UHFMAN_SELECT_MODE_NEVER:
            uMode = YPDR200_X12_MODE_NEVER;
            break;
        case UHFMAN_SELECT_MODE_RWLK:
            uMode = YPDR200_X12_MODE_RWLK;
            break;
        default:
            LOG_W("Unsupported select mode: %d", mode);
            return UHFMAN_SET_SELECT_MODE_ERR_NOT_SUPPORTED;
    }
    ypdr200_resp_err_code_t rerr = 0;
    int rv = ypdr200_x12(pCtx, uMode, &rerr);
    switch (rv) {
        case YPDR200_X0E_ERR_SUCCESS:
            uhfman_config_select_mode_update(pCtx, mode);
            return UHFMAN_SET_SELECT_MODE_ERR_SUCCESS;
        case YPDR200_X0E_ERR_SEND_COMMAND:
            return UHFMAN_SET_SELECT_MODE_ERR_SEND_COMMAND;
        case YPDR200_X0E_ERR_READ_RESPONSE:
            return UHFMAN_SET_SELECT_MODE_ERR_READ_RESPONSE;
        case YPDR200_X0E_ERR_ERROR_RESPONSE:
            LOG_W("** Response frame was an error frame containing error code 0x%02X **", (uint8_t)rerr);
            return UHFMAN_SET_SELECT_MODE_ERR_ERROR_RESPONSE;
        default:
            LOG_W("Unknown error from ypdr200_x12: %d", rv);
            return UHFMAN_SET_SELECT_MODE_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_SET_SELECT_MODE_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}

uhfman_err_t uhfman_dbg_get_query_params(uhfman_ctx_t* pCtx) {
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    ypdr200_resp_err_code_t rerr = 0;
    ypdr200_x0d_resp_param_t respParam;
    int rv = ypdr200_x0d(pCtx, &respParam, &rerr);
    switch (rv) {
        case YPDR200_X0D_ERR_SUCCESS:
            LOG_I("Query parameters: dr=0x%02X, m=0x%02X, trext=0x%02X, sel=0x%02X, session=0x%02X, target=0x%02X q=0x%02X", respParam.dr, respParam.m, respParam.trext, respParam.sel, respParam.session, respParam.target, respParam.q);
            return UHFMAN_GET_QUERY_PARAMS_ERR_SUCCESS;
        case YPDR200_X0D_ERR_SEND_COMMAND:
            return UHFMAN_GET_QUERY_PARAMS_ERR_SEND_COMMAND;
        case YPDR200_X0D_ERR_READ_RESPONSE:
            return UHFMAN_GET_QUERY_PARAMS_ERR_READ_RESPONSE;
        case YPDR200_X0D_ERR_ERROR_RESPONSE:
            LOG_W("** Response frame was an error frame containing error code 0x%02X **", (uint8_t)rerr);
            return UHFMAN_GET_QUERY_PARAMS_ERR_ERROR_RESPONSE;
        default:
            LOG_W("Unknown error from ypdr200_x0d: %d", rv);
            return UHFMAN_GET_QUERY_PARAMS_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_GET_QUERY_PARAMS_ERR_UNKNOWN_DEVICE_MODEL; // TODO Change to #error or remove, because generating the error once is enough (applies to other functions here as well)
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}

static void uhfman_config_query_params_update(uhfman_ctx_t* pCtx, uhfman_query_sel_t sel, uhfman_query_session_t session, uhfman_query_target_t target, uint8_t q) {
    pCtx->_config.query_params.sel = sel;
    pCtx->_config.query_params.session = session;
    pCtx->_config.query_params.target = target;
    pCtx->_config.query_params.q = q;
    pCtx->_config.flags |= UHFMAN_CTX_CONFIG_FLAG_QUERY_INITIALIZED;
}

static int uhfman_config_query_params_cache_cmp(uhfman_ctx_t* pCtx, uhfman_query_sel_t sel, uhfman_query_session_t session, uhfman_query_target_t target, uint8_t q) {
    if ((pCtx->_config.flags & UHFMAN_CTX_CONFIG_FLAG_QUERY_INITIALIZED) == 0) {
        return 1;
    }
    return (pCtx->_config.query_params.sel != sel)
    || (pCtx->_config.query_params.session != session)
    || (pCtx->_config.query_params.target != target)
    || (pCtx->_config.query_params.q != q);
}

uhfman_err_t uhfman_set_query_params(uhfman_ctx_t* pCtx, uhfman_query_sel_t sel, uhfman_query_session_t session, uhfman_query_target_t target, uint8_t q) {
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    if (!uhfman_config_query_params_cache_cmp(pCtx, sel, session, target, q)) {
        return UHFMAN_SET_QUERY_PARAMS_ERR_SUCCESS;
    }
    ypdr200_x0e_req_param_t reqParam = {
        .dr = 0x00, // DR=8
        .m = 0x00, // M=1
        .trext = 0x01, // Use pilot tone
        .q = q,
        .reserved = 0x00
    };
    switch (sel) {
        case UHFMAN_QUERY_SEL_ALL:
            reqParam.sel = 0x00;
            break;
        case UHFMAN_QUERY_SEL_ALL1:
            reqParam.sel = 0x01;
            break;
        case UHFMAN_QUERY_SEL_NOT_SL:
            reqParam.sel = 0x02;
            break;
        case UHFMAN_QUERY_SEL_SL:
            reqParam.sel = 0x03;
            break;
        default:
            LOG_W("Unsupported query sel field value: %d", sel);
            return UHFMAN_SET_QUERY_PARAMS_ERR_NOT_SUPPORTED;
    }
    switch (session) {
        case UHFMAN_QUERY_SESSION_S0:
            reqParam.session = 0x00;
            break;
        case UHFMAN_QUERY_SESSION_S1:
            reqParam.session = 0x01;
            break;
        case UHFMAN_QUERY_SESSION_S2:
            reqParam.session = 0x02;
            break;
        case UHFMAN_QUERY_SESSION_S3:
            reqParam.session = 0x03;
            break;
        default:
            LOG_W("Unsupported query session field value: %d", session);
            return UHFMAN_SET_QUERY_PARAMS_ERR_NOT_SUPPORTED;
    }
    ypdr200_resp_err_code_t rerr = 0;
    int rv = ypdr200_x0e(pCtx, reqParam, &rerr);
    switch (rv) {
        case YPDR200_X0E_ERR_SUCCESS:
            uhfman_config_query_params_update(pCtx, sel, session, target, q);
            return UHFMAN_SET_QUERY_PARAMS_ERR_SUCCESS;
        case YPDR200_X0E_ERR_SEND_COMMAND:
            return UHFMAN_SET_QUERY_PARAMS_ERR_SEND_COMMAND;
        case YPDR200_X0E_ERR_READ_RESPONSE:
            return UHFMAN_SET_QUERY_PARAMS_ERR_READ_RESPONSE;
        case YPDR200_X0E_ERR_ERROR_RESPONSE:
            LOG_W("** Response frame was an error frame containing error code 0x%02X **", (uint8_t)rerr);
            return UHFMAN_SET_QUERY_PARAMS_ERR_ERROR_RESPONSE;
        default:
            LOG_W("Unknown error from ypdr200_x0e: %d", rv);
            return UHFMAN_SET_QUERY_PARAMS_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_SET_QUERY_PARAMS_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}

uhfman_err_t uhfman_dbg_get_working_channel(uhfman_ctx_t* pCtx) {
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    ypdr200_resp_err_code_t rerr = 0;
    uint8_t chIndex = 0;
    int rv = ypdr200_xaa(pCtx, &chIndex, &rerr);
    switch (rv) {
        case YPDR200_XAA_ERR_SUCCESS:
            LOG_I("Working channel: 0x%02X", chIndex);
            return UHFMAN_GET_WORKING_CHANNEL_ERR_SUCCESS;
        case YPDR200_XAA_ERR_SEND_COMMAND:
            return UHFMAN_GET_WORKING_CHANNEL_ERR_SEND_COMMAND;
        case YPDR200_XAA_ERR_READ_RESPONSE:
            return UHFMAN_GET_WORKING_CHANNEL_ERR_READ_RESPONSE;
        case YPDR200_XAA_ERR_ERROR_RESPONSE:
            LOG_W("** Response frame was an error frame containing error code 0x%02X **", (uint8_t)rerr);
            return UHFMAN_GET_WORKING_CHANNEL_ERR_ERROR_RESPONSE;
        default:
            LOG_W("Unknown error from ypdr200_xaa: %d", rv);
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
            LOG_I("Work area/Region: 0x%02X", region);
            return UHFMAN_GET_WORK_AREA_ERR_SUCCESS;
        case YPDR200_X08_ERR_SEND_COMMAND:
            return UHFMAN_GET_WORK_AREA_ERR_SEND_COMMAND;
        case YPDR200_X08_ERR_READ_RESPONSE:
            return UHFMAN_GET_WORK_AREA_ERR_READ_RESPONSE;
        case YPDR200_X08_ERR_ERROR_RESPONSE:
            LOG_W("** Response frame was an error frame containing error code 0x%02X **", (uint8_t)rerr);
            return UHFMAN_GET_WORK_AREA_ERR_ERROR_RESPONSE;
        default:
            LOG_W("Unknown error from ypdr200_x08: %d", rv);
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
            float dbm = ((float)powerLevel) / 100.0f;
            LOG_I("Transmit power level: 0x%02X (%2.2f dBm)", powerLevel, dbm);
            return UHFMAN_GET_TRANSMIT_POWER_ERR_SUCCESS;
        case YPDR200_XB7_ERR_SEND_COMMAND:
            return UHFMAN_GET_TRANSMIT_POWER_ERR_SEND_COMMAND;
        case YPDR200_XB7_ERR_READ_RESPONSE:
            return UHFMAN_GET_TRANSMIT_POWER_ERR_READ_RESPONSE;
        case YPDR200_XB7_ERR_ERROR_RESPONSE:
            LOG_W("** Response frame was an error frame containing error code 0x%02X **", (uint8_t)rerr);
            return UHFMAN_GET_TRANSMIT_POWER_ERR_ERROR_RESPONSE;
        default:
            LOG_W("Unknown error from ypdr200_xb7: %d", rv);
            return UHFMAN_GET_TRANSMIT_POWER_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_GET_TRANSMIT_POWER_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}

static void uhfman_config_tx_power_update(uhfman_ctx_t* pCtx, float txPower) {
    pCtx->_config.txPower = txPower;
    pCtx->_config.flags |= UHFMAN_CTX_CONFIG_FLAG_TX_POWER_INITIALIZED;
}

static int uhfman_config_tx_power_cache_cmp(uhfman_ctx_t* pCtx, float txPower) {
    if ((pCtx->_config.flags & UHFMAN_CTX_CONFIG_FLAG_TX_POWER_INITIALIZED) == 0) {
        return 1;
    }
    return (pCtx->_config.txPower != txPower) ? 0 : 1;
}

uhfman_err_t uhfman_set_transmit_power(uhfman_ctx_t* pCtx, float txPower) {
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    if (!uhfman_config_tx_power_cache_cmp(pCtx, txPower)) {
        LOG_I("Transmit power level already set to: %2.2f dBm", txPower);
        return UHFMAN_SET_TRANSMIT_POWER_ERR_SUCCESS;
    }
    ypdr200_resp_err_code_t rerr = 0;
    uint16_t powerLevel = (uint16_t)(txPower * 100.0f);
    int rv = ypdr200_xb6(pCtx, powerLevel, &rerr);
    switch (rv) {
        case YPDR200_XB6_ERR_SUCCESS:
            LOG_I("Transmit power level set to: 0x%02X (%2.2f dBm)", powerLevel, txPower);
            uhfman_config_tx_power_update(pCtx, txPower);
            return UHFMAN_SET_TRANSMIT_POWER_ERR_SUCCESS;
        case YPDR200_XB6_ERR_SEND_COMMAND:
            return UHFMAN_SET_TRANSMIT_POWER_ERR_SEND_COMMAND;
        case YPDR200_XB6_ERR_READ_RESPONSE:
            return UHFMAN_SET_TRANSMIT_POWER_ERR_READ_RESPONSE;
        case YPDR200_XB6_ERR_ERROR_RESPONSE:
            LOG_W("** Response frame was an error frame containing error code 0x%02X **", (uint8_t)rerr);
            return UHFMAN_SET_TRANSMIT_POWER_ERR_ERROR_RESPONSE;
        default:
            LOG_W("Unknown error from ypdr200_xb6: %d", rv);
            return UHFMAN_SET_TRANSMIT_POWER_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_SET_TRANSMIT_POWER_ERR_UNKNOWN_DEVICE_MODEL;
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
            LOG_I("Demodulator parameters: mixer_G: 0x%02X, if_G: 0x%02X, thrdMsb: 0x%02X, thrdLsb: 0x%02X (thrd: 0x%04X)", respParam.mixer_G, respParam.if_G, respParam.thrdMsb, respParam.thrdLsb, thrd);
            return UHFMAN_GET_DEMOD_PARAMS_ERR_SUCCESS;
        case YPDR200_XF1_ERR_SEND_COMMAND:
            return UHFMAN_GET_DEMOD_PARAMS_ERR_SEND_COMMAND;
        case YPDR200_XF1_ERR_READ_RESPONSE:
            return UHFMAN_GET_DEMOD_PARAMS_ERR_READ_RESPONSE;
        case YPDR200_XF1_ERR_ERROR_RESPONSE:
            LOG_W("** Response frame was an error frame containing error code 0x%02X **", (uint8_t)rerr);
            return UHFMAN_GET_DEMOD_PARAMS_ERR_ERROR_RESPONSE;
        default:
            LOG_W("Unknown error from ypdr200_xf1: %d", rv);
            return UHFMAN_GET_DEMOD_PARAMS_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_GET_DEMOD_PARAMS_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}

static uhfman_tag_t __uhfman_tags[UHFMAN_MAX_NUM_TAGS];
static uint32_t __num_tags = 0U;
static uhfman_tag_handler_t __uhfman_new_tag_handler = NULL;
static uhfman_poll_handler_t __uhfman_poll_handler = NULL;

#define UHFMAN_TAG_HANDLE_INVALID 0xFFFFU

static uint16_t __uhfman_tag_find(uint8_t* epc) { // TODO Optimize this search
    for (uint32_t i = 0; i < __num_tags; i++) {
        if (memcmp(__uhfman_tags[i].epc, epc, YPDR200_X22_NTF_PARAM_EPC_LENGTH) == 0) {
            //fprintf(stdout, "handle=%u\n", (uint16_t)i);
            return (uint16_t)i;
        }
    }
    return (uint16_t)UHFMAN_TAG_HANDLE_INVALID;
}

static void __uhfman_tag_add_read(uint8_t* epc, uint8_t rssi) { // TODO think about splitting into threads if performance is an issue?
    if (epc == NULL) {
        assert(0);
        return;
    }

    struct timeval tv;
    gettimeofday(&tv,NULL);
    uint32_t time_ms = 1000 * tv.tv_sec + (tv.tv_usec / 1000);
    
    uint16_t tagHandle = __uhfman_tag_find(epc);
    if (tagHandle != UHFMAN_TAG_HANDLE_INVALID) {
        uint32_t readMod = __uhfman_tags[tagHandle].num_reads % UHFMAN_TAG_PERIOD_NREADS; // write in a circular buffer fashion
        __uhfman_tags[tagHandle].read_times[readMod] = time_ms;
        //printf("readMod: %d, read_times[readMod]: %lu\n", readMod, __uhfman_tags[tagHandle].read_times[readMod]);
        __uhfman_tags[tagHandle].rssi[readMod] = rssi;
        __uhfman_tags[tagHandle].num_reads++;
        if (__uhfman_poll_handler != NULL) {
            __uhfman_poll_handler(tagHandle);
        }
        return;
    }

    if (__num_tags >= UHFMAN_MAX_NUM_TAGS) {
        assert(0);
        return;
    } else {
        memcpy(__uhfman_tags[__num_tags].epc, epc, YPDR200_X22_NTF_PARAM_EPC_LENGTH);
        __uhfman_tags[__num_tags].rssi[0] = rssi;
        __uhfman_tags[__num_tags].read_times[0] = time_ms;
        __uhfman_tags[__num_tags].num_reads = 1;
        __uhfman_tags[__num_tags].handle = __num_tags;
        __num_tags++;
        if (__uhfman_new_tag_handler != NULL) {
            __uhfman_new_tag_handler(__uhfman_tags[__num_tags - 1]);
        }
        if (__uhfman_poll_handler != NULL) {
            __uhfman_poll_handler(__num_tags - 1); // TODO change when refactoring to using multiple threads?
        }
    }
}

#if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
static void uhfman_dbg_single_polling_notification_handler(ypdr200_x22_ntf_param_t ntfParam, const void* pUserData) {
    // if ((ntfParam.crc[0] == 0x07 && ntfParam.crc[1] == 0x75)
    //     || (ntfParam.crc[0] == 0xDC && ntfParam.crc[1] == 0x20)) {
    //     return; // Ignore those tags
    // }
    LOG_D_TBC("Polling notification: rssi=0x%02X, pc=0x%04X, epc=[ ", ntfParam.rssi, ((uint16_t)(ntfParam.pc[0]) << 8) | (uint16_t)(ntfParam.pc[1]));
    for (int i = 0; i < YPDR200_X22_NTF_PARAM_EPC_LENGTH; i++) { //TODO variabilize (related to #ae3759b4)
        LOG_D_CTBC("%02X ", ntfParam.epc[i]);
    }
    LOG_D_CFIN("], crc=0x%04X", ((uint16_t)(ntfParam.crc[0]) << 8) | (uint16_t)(ntfParam.crc[1]));
    __uhfman_tag_add_read(ntfParam.epc, ntfParam.rssi);
}
#else
#error "Unknown UHFMAN_DEVICE_MODEL"
#endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200

void uhfman_list_tags(uhfman_tag_t** ppTags, uint32_t* pnTags_out) {
    *ppTags = (uhfman_tag_t*) malloc(__num_tags * sizeof(uhfman_tag_t));
    if (*ppTags == NULL) {
        errno = ENOMEM;
        return;
    }
    memcpy(*ppTags, __uhfman_tags, __num_tags * sizeof(uhfman_tag_t));
    *pnTags_out = __num_tags;
}

void uhfman_set_new_tag_event_handler(uhfman_tag_handler_t handler) {
    __uhfman_new_tag_handler = handler;
}

void uhfman_unset_new_tag_event_handler() {
    __uhfman_new_tag_handler = NULL;
}

void ufhman_set_poll_handler(uhfman_poll_handler_t handler) {
    __uhfman_poll_handler = handler;
}

void ufhman_unset_poll_handler() {
    __uhfman_poll_handler = NULL;
}

uhfman_tag_t uhfman_tag_get(uint16_t handle) {
    return __uhfman_tags[handle];
}

uhfman_tag_stats_t uhfman_tag_get_stats(uint16_t handle) {
    uhfman_tag_t tag = __uhfman_tags[handle];

    uint32_t rssi_sum = 0U;
    uint32_t iMax = tag.num_reads < UHFMAN_TAG_PERIOD_NREADS ? tag.num_reads : UHFMAN_TAG_PERIOD_NREADS;
    if (iMax < 2) {
        return (uhfman_tag_stats_t) {
            .rssi_avg_per_period = NAN,
            .read_time_interval_avg_per_period = -1U
        };
    }
    for (uint32_t i = 0; i < iMax; i++) {
        rssi_sum += tag.rssi[i];
    }

    //printf("( "); //dbg
    uint32_t read_interval_sum = 0U; //TODO support less than 32-bit systems
    int difference = ((int)(tag.read_times[0])) - ((int)(tag.read_times[iMax - 1]));
    if (difference > 0) {
        read_interval_sum += difference;
        //printf("%u-%u=%d ", tag.read_times[0], tag.read_times[iMax - 1], difference); //dbg
    }
    for (uint32_t i = 0; i < iMax - 1; i++) {
        difference = ((int)(tag.read_times[i + 1])) - ((int)(tag.read_times[i]));
        if (difference > 0) { // TODO optimize using modulo instead of this condition in loop
            read_interval_sum += difference;
            //printf("%u-%u=%d ", tag.read_times[i + 1], tag.read_times[i], difference); //dbg
        }
    }
    //printf(")\n");
    

    //printf("rssi_sum: %d, iMax: %d, read_interval_sum: %lu\n", rssi_sum, iMax, read_interval_sum);

    return (uhfman_tag_stats_t) {
        .rssi_avg_per_period = ((float)rssi_sum) / (float)(iMax),
        .read_time_interval_avg_per_period = read_interval_sum / (iMax - 1)
    };
}

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
            LOG_W("** Response frame was an error frame containing error code 0x%02X **", (uint8_t)rerr);
            return UHFMAN_SINGLE_POLLING_ERR_ERROR_RESPONSE;
        case YPDR200_X22_ERR_READ_NOTIFICATION:
            return UHFMAN_SINGLE_POLLING_ERR_READ_NOTIFICATION;
        case YPDR200_X22_ERR_UNEXPECTED_FRAME_TYPE:
            return UHFMAN_SINGLE_POLLING_ERR_UNEXPECTED_FRAME_TYPE;
        default:
            LOG_W("Unknown error from ypdr200_x22: %d", rv);
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
            LOG_W("** Response frame was an error frame containing error code 0x%02X **", (uint8_t)rerr);
            return UHFMAN_SINGLE_POLLING_ERR_ERROR_RESPONSE;
        case YPDR200_X27_ERR_READ_NOTIFICATION:
            return UHFMAN_SINGLE_POLLING_ERR_READ_NOTIFICATION;
        case YPDR200_X27_ERR_UNEXPECTED_FRAME_TYPE:
            return UHFMAN_SINGLE_POLLING_ERR_UNEXPECTED_FRAME_TYPE;
        default:
            LOG_W("Unknown error from ypdr200_x27: %d", rv);
            return UHFMAN_SINGLE_POLLING_ERR_UNKNOWN;
    }
    #else
    return UHFMAN_DBG_SINGLE_POLLING_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}

#define __UHFMAN_TAG_PC_LENGTH 2
// @[N_267abdf1] This may need to be loosened in the future version
#define __UHFMAN_TAG_EPC_LENGTH UHFMAN_TAG_EPC_STANDARD_LENGTH

uhfman_err_t uhfman_write_tag_mem(uhfman_ctx_t* pCtx, 
                                  const uint8_t accessPasswd[4], 
                                  uhfman_tag_mem_bank_t memBank, 
                                  uint16_t wordPtr, 
                                  uint16_t nWords, 
                                  const uint8_t* pData,
                                  uint16_t* pPC_out,
                                  uint8_t** ppEPC_out,
                                  size_t* pEPC_len_out,
                                  uint8_t* pRespErrCode_out) {
    if (pRespErrCode_out != NULL) {
        *pRespErrCode_out = 0U; // assume success frame err code until we receive an error response frame with an error code (we do not won't garbage value in here)
    }
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    if (accessPasswd[0] == 0 && accessPasswd[1] == 0 && accessPasswd[2] == 0 && accessPasswd[3] == 0) {
        //LOG_W("Access password is all zeros, which is not allowed");
        //return UHFMAN_WRITE_TAG_MEM_ERR_NOT_SUPPORTED;
        LOG_W("Access password is all zeros");
    }
    //use constructing functions to create a ypdr200_x49_req_param_t instance
    uint8_t uMemBank = 0xFF;
    switch (memBank) {
        case UHFMAN_TAG_MEM_BANK_RESERVED:
            uMemBank = 0x00;
            break;
        case UHFMAN_TAG_MEM_BANK_EPC:
            uMemBank = 0x01;
            break;
        case UHFMAN_TAG_MEM_BANK_TID:
            uMemBank = 0x02;
            break;
        case UHFMAN_TAG_MEM_BANK_USER:
            uMemBank = 0x03;
            break;
        default:
            LOG_W("Unsupported memory bank: %d", memBank);
            return UHFMAN_WRITE_TAG_MEM_ERR_NOT_SUPPORTED;
    }
    assert(uMemBank != 0xFF); //defensive
    ypdr200_x49_req_param_hdr_t reqParamHdr = ypdr200_x49_req_param_hdr_make(wordPtr, nWords, uMemBank, accessPasswd);
    ypdr200_x49_req_param_t reqParam = ypdr200_x49_req_param_make(reqParamHdr, (uint8_t*)pData); 

    ypdr200_resp_err_code_t rerr = 0;
    ypdr200_x49_resp_param_t respParam = {};
    LOG_I_TBC("Writing to tag memory: accessPasswd=[ %02X %02X %02X %02X ], memBank=0x%02X, wordPtr=0x%04X, nWords=0x%04X, data=[ ", accessPasswd[0], accessPasswd[1], accessPasswd[2], accessPasswd[3], uMemBank, wordPtr, nWords);
    for (int i = 0; i < (nWords << 1); i++) {
        LOG_I_CTBC("%02X ", pData[i]);
    }
    LOG_I_CFIN("]");
    int rv = ypdr200_x49(pCtx, reqParam, &respParam, &rerr);
    switch (rv) {
        case YPDR200_X49_ERR_SUCCESS:
            assert(respParam.ul == __UHFMAN_TAG_EPC_LENGTH + __UHFMAN_TAG_PC_LENGTH); //defensive + see @[N_267abdf1]
            if (pPC_out != NULL) {
                *pPC_out = ((uint16_t)(respParam.pc[0]) << 8) | (uint16_t)(respParam.pc[1]);
            }
            if (ppEPC_out != NULL) {
                *ppEPC_out = (uint8_t*) malloc(__UHFMAN_TAG_EPC_LENGTH * sizeof(uint8_t));
                memcpy(*ppEPC_out, respParam.epc, __UHFMAN_TAG_EPC_LENGTH * sizeof(uint8_t));
            }
            if (pEPC_len_out != NULL) {
                *pEPC_len_out = __UHFMAN_TAG_EPC_LENGTH;
            }
            return UHFMAN_WRITE_TAG_MEM_ERR_SUCCESS;
        case YPDR200_X49_ERR_SEND_COMMAND:
            return UHFMAN_WRITE_TAG_MEM_ERR_SEND_COMMAND;
        case YPDR200_X49_ERR_READ_RESPONSE:
            return UHFMAN_WRITE_TAG_MEM_ERR_READ_RESPONSE;
        case YPDR200_X49_ERR_ERROR_RESPONSE:
            LOG_W("** Response frame was an error frame containing error code 0x%02X **", (uint8_t)rerr);
            if (pPC_out != NULL) {
                *pPC_out = ((uint16_t)(respParam.pc[0]) << 8) | (uint16_t)(respParam.pc[1]);
            }
            if (ppEPC_out != NULL) {
                *ppEPC_out = (uint8_t*) malloc(__UHFMAN_TAG_EPC_LENGTH * sizeof(uint8_t));
                memcpy(*ppEPC_out, respParam.epc, __UHFMAN_TAG_EPC_LENGTH * sizeof(uint8_t));
            }
            if (pEPC_len_out != NULL) {
                *pEPC_len_out = __UHFMAN_TAG_EPC_LENGTH;
            }
            if (pRespErrCode_out != NULL) {
                switch (rerr) {
                    case YPDR200_RESP_ERR_CODE_ACCESS_FAIL:
                        *pRespErrCode_out = UHFMAN_TAG_ERR_ACCESS_DENIED;
                        break;
                    default:
                        *pRespErrCode_out = UHFMAN_TAG_ERR_UNKNOWN;
                        break;
                }
            }
            return UHFMAN_WRITE_TAG_MEM_ERR_ERROR_RESPONSE;
        default:
            LOG_W("Unknown error from ypdr200_x49: %d", rv);
            return UHFMAN_WRITE_TAG_MEM_ERR_UNKNOWN;
    }    
    #else
    return UHFMAN_WRITE_TAG_MEM_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200   
}

uhfman_err_t uhfman_lock_tag_mem(uhfman_ctx_t* pCtx, 
                                 const uint8_t accessPasswd[4],
                                 uint16_t lock_mask_flags, 
                                 uint16_t lock_action_flags,
                                 uint16_t* pPC_out,
                                 uint8_t** ppEPC_out,
                                 size_t* pEPC_len_out,
                                 uint8_t* pRespErrCode_out) {
    if (pRespErrCode_out != NULL) {
        *pRespErrCode_out = 0U; // assume success frame err code until we receive an error response frame with an error code (we do not won't garbage value in here)
    }
    #if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
    if (accessPasswd[0] == 0 && accessPasswd[1] == 0 && accessPasswd[2] == 0 && accessPasswd[3] == 0) {
        //LOG_W("Access password is all zeros, which is not allowed");
        //return UHFMAN_LOCK_TAG_MEM_ERR_NOT_SUPPORTED;
        LOG_W("Access password is all zeros");
    }
    ypdr200_x82_req_param_t reqParam = {};
    memcpy(reqParam.ap, accessPasswd, 4 * sizeof(uint8_t));
    memset(reqParam.ld, 0, 3 * sizeof(uint8_t));
    //ld = 0000xxxx xxxxxxxx xxxxxxxx
    reqParam.ld[0] &= 0x0F; // First 4 bits are reserved
    //             ld = 0000VVVV xxxxxxxx xxxxxxxx
    //lock_mask_flags = 000000VV VV000000
    reqParam.ld[0] |= (lock_mask_flags >> 6) & 0x0F; 
    //             ld = 0000xxxx VVVVVVxx xxxxxxxx
    //lock_mask_flags = 000000xx xxVVVVVV
    reqParam.ld[1] |= (lock_mask_flags & 0x3F) << 2;
    //               ld = 0000xxxx xxxxxxVV xxxxxxxx
    //lock_action_flags = 000000VV xxxxxxxx
    reqParam.ld[1] |= (lock_action_flags >> 8) & 0x03;
    //               ld = 0000xxxx xxxxxxxx VVVVVVVV
    //lock_action_flags = 000000xx VVVVVVVV
    reqParam.ld[2] |= lock_action_flags & 0xFF;
    ypdr200_resp_err_code_t rerr = 0;
    ypdr200_x82_resp_param_t respParam = {};
    LOG_I_TBC("Locking tag memory: accessPasswd=[ %02X %02X %02X %02X ], lock_mask_flags=0x%04X, lock_action_flags=0x%04X", accessPasswd[0], accessPasswd[1], accessPasswd[2], accessPasswd[3], lock_mask_flags, lock_action_flags);
    int rv = ypdr200_x82(pCtx, reqParam, &respParam, &rerr);
    switch (rv) {
        case YPDR200_X82_ERR_SUCCESS:
            assert(respParam.ul == __UHFMAN_TAG_EPC_LENGTH + __UHFMAN_TAG_PC_LENGTH); //defensive + see @[N_267abdf1]
            if (pPC_out != NULL) {
                *pPC_out = ((uint16_t)(respParam.pc[0]) << 8) | (uint16_t)(respParam.pc[1]);
            }
            if (ppEPC_out != NULL) {
                *ppEPC_out = (uint8_t*) malloc(__UHFMAN_TAG_EPC_LENGTH * sizeof(uint8_t));
                memcpy(*ppEPC_out, respParam.epc, __UHFMAN_TAG_EPC_LENGTH * sizeof(uint8_t));
            }
            if (pEPC_len_out != NULL) {
                *pEPC_len_out = __UHFMAN_TAG_EPC_LENGTH;
            }
            return UHFMAN_LOCK_TAG_MEM_ERR_SUCCESS;
        case YPDR200_X82_ERR_SEND_COMMAND:
            return UHFMAN_LOCK_TAG_MEM_ERR_SEND_COMMAND;
        case YPDR200_X82_ERR_READ_RESPONSE:
            return UHFMAN_LOCK_TAG_MEM_ERR_READ_RESPONSE;
        case YPDR200_X82_ERR_ERROR_RESPONSE:
            LOG_W("** Response frame was an error frame containing error code 0x%02X **", (uint8_t)rerr);
            if (pPC_out != NULL) {
                *pPC_out = ((uint16_t)(respParam.pc[0]) << 8) | (uint16_t)(respParam.pc[1]);
            }
            if (ppEPC_out != NULL) {
                *ppEPC_out = (uint8_t*) malloc(__UHFMAN_TAG_EPC_LENGTH * sizeof(uint8_t));
                memcpy(*ppEPC_out, respParam.epc, __UHFMAN_TAG_EPC_LENGTH * sizeof(uint8_t));
            }
            if (pEPC_len_out != NULL) {
                *pEPC_len_out = __UHFMAN_TAG_EPC_LENGTH;
            }
            if (pRespErrCode_out != NULL) {
                switch (rerr) {
                    case YPDR200_RESP_ERR_CODE_ACCESS_FAIL:
                        *pRespErrCode_out = UHFMAN_TAG_ERR_ACCESS_DENIED;
                        break;
                    default:
                        *pRespErrCode_out = UHFMAN_TAG_ERR_UNKNOWN;
                        break;
                }
            }
            return UHFMAN_WRITE_TAG_MEM_ERR_ERROR_RESPONSE;
        default:
            LOG_W("Unknown error from ypdr200_x82: %d", rv);
            return UHFMAN_WRITE_TAG_MEM_ERR_UNKNOWN;
    }    
    #else
    return UHFMAN_LOCK_TAG_MEM_ERR_UNKNOWN_DEVICE_MODEL;
    #endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
}