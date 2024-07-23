#include "uhfman_common.h"

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

/**
 * @brief Get reader/writer module information
 * @param infoType Type of information to be retrieved
 */
int ypdr200_x03(uhfman_ctx_t* pCtx, ypdr200_x03_param_t infoType, char** ppcInfo_out);
