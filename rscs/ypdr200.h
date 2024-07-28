#ifndef YPDR200_H
#define YPDR200_H

#include "uhfman_common.h"

#ifndef YPDR200_INTERFACE_TYPE
#error "YPDR200_INTERFACE_TYPE is not defined"
#endif

// typedef struct {
//     libusb_device_handle *handle;
//     libusb_context *context;
//     struct ypdr200_config {
//         // Nothing here for now
//     } config;
// } ypdr200_ctx_t;

typedef enum {
    YPDR200_X11_PARAM_BAUD_RATE_115200 = 115200U
} ypdr200_x11_param_t;

#define YPDR200_X11_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_X11_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
/**
 * @brief Set baud rate for communication with the YPD-R200 reader/writer module
 * @param baudRate Baud rate to be set
 * @note It seems like the default baud rate supported by the device is 115200 bps - you may not need to call this function if you want to use the default baud rate
 * @attention There are communication issues after using this function - please stick with the default baud rate if possible (or contribute to fix this issue). Btw. this command was hidden in the v2.3.3 user protocol document after the "kill tag" command specification - maybe it's not supported by the device actually and it was a copy-paste mistake by the document author? 
 */
int ypdr200_x11(uhfman_ctx_t* pCtx, ypdr200_x11_param_t baudRate);

typedef enum {
    YPDR200_X03_PARAM_HARDWARE_VERSION,
    YPDR200_X03_PARAM_SOFTWARE_VERSION,
    YPDR200_X03_PARAM_MANUFACTURER
} ypdr200_x03_param_t;

#define YPDR200_X03_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_X03_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_X03_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
/**
 * @brief Get reader/writer module information
 * @param infoType Type of information to be retrieved
 */
int ypdr200_x03(uhfman_ctx_t* pCtx, ypdr200_x03_param_t infoType, char** ppcInfo_out);

#endif // YPDR200_H