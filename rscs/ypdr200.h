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