#ifndef YPDR200_H
#define YPDR200_H

#include "uhfman_common.h"

#ifndef YPDR200_INTERFACE_TYPE
#error "YPDR200_INTERFACE_TYPE is not defined"
#endif

typedef enum {
    YPDR200_RESP_ERR_CODE_COMMAND_ERROR = 0x17,
    YPDR200_RESP_ERR_CODE_FHSS_FAIL = 0x20,
    YPDR200_RESP_ERR_CODE_INVENTORY_FAIL = 0x15,
    YPDR200_RESP_ERR_CODE_ACCESS_FAIL = 0x16,
    YPDR200_RESP_ERR_CODE_READ_FAIL = 0x09,
    YPDR200_RESP_ERR_CODE_READ_ERROR_MASK = 0xA0,
    YPDR200_RESP_ERR_CODE_WRITE_FAIL = 0x10,
    YPDR200_RESP_ERR_CODE_WRITE_ERROR_MASK = 0xB0,
    YPDR200_RESP_ERR_CODE_LOCK_FAIL = 0x13,
    YPDR200_RESP_ERR_CODE_LOCK_ERROR_MASK = 0xC0,
    YPDR200_RESP_ERR_CODE_KILL_FAIL = 0x12,
    YPDR200_RESP_ERR_CODE_KILL_ERROR_MASK = 0xD0,
    YPDR200_RESP_ERR_CODE_BLOCK_PERMALOCK_FAIL = 0x14,
    YPDR200_RESP_ERR_CODE_BLOCK_PERMALOCK_ERROR_MASK = 0xE0,

    /* NXP G2X specific */
    YPDR200_RESP_ERR_CODE_NXP_G2X_CHANGE_CONFIG_FAIL = 0x1A,
    YPDR200_RESP_ERR_CODE_NXP_G2X_READ_PROTECT_FAIL = 0x2A,
    YPDR200_RESP_ERR_CODE_NXP_G2X_RESET_READ_PROTECT_FAIL = 0x2B,
    YPDR200_RESP_ERR_CODE_NXP_G2X_CHANGE_EAS_FAIL = 0x1B,
    YPDR200_RESP_ERR_CODE_NXP_G2X_EAS_ALARM_FAIL = 0x1D,
    YPDR200_RESP_ERR_CODE_NXP_G2X_SPECIAL_FLAG = 0xE0,

    /* Impinj Monza Qt specific */
    YPDR200_RESP_ERR_CODE_IMPINJ_MONZA_QT_SPECIAL_MASK = 0xE0,

    /* EPC Gen2 protocol tag error codes */
    YPDR200_RESP_ERR_CODE_TAGEPCGEN2_OTHER_ERROR_MASK = 0b00000000,
    YPDR200_RESP_ERR_CODE_TAGEPCGEN2_MEMORY_OVERRUN_MASK = 0b00000011,
    YPDR200_RESP_ERR_CODE_TAGEPCGEN2_MEMORY_LOCKED_MASK = 0b00000100,
    YPDR200_RESP_ERR_CODE_TAGEPCGEN2_INSUFFICIENT_POWER_MASK = 0b00001011,
    YPDR200_RESP_ERR_CODE_TAGEPCGEN2_NON_SPECIFIC = 0b00001111
} ypdr200_resp_err_code_t;

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
} ypdr200_x03_req_param_t;

#define YPDR200_X03_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_X03_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_X03_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define YPDR200_X03_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
/**
 * @brief Get reader/writer module information
 * @param infoType Type of information to be retrieved
 */
int ypdr200_x03(uhfman_ctx_t* pCtx, ypdr200_x03_req_param_t infoType, char** ppcInfo_out, ypdr200_resp_err_code_t* pRespErrCode);


#define YPDR200_X0B_RESP_PARAM_HDR_SIZE 7
typedef union {
    struct {
        union {
            struct {
                #if defined(TARGET_LITTLE_ENDIAN)
                uint8_t memBank : 2;
                uint8_t action  : 3;
                uint8_t target  : 3;
                #elif defined(TARGET_BIG_ENDIAN)
                uint8_t target  : 3;
                uint8_t action  : 3;
                uint8_t memBank : 2;
                #else
                #error "Neither TARGET_LITTLE_ENDIAN nor TARGET_BIG_ENDIAN is defined"
                #endif
            };
            uint8_t selParam;
        };
        uint8_t ptr[4]; //MSB first
        uint8_t maskLen;
        uint8_t truncate;
    };
    uint8_t raw[YPDR200_X0B_RESP_PARAM_HDR_SIZE];
} __attribute__((__packed__)) ypdr200_x0b_resp_param_hdr_t;

typedef struct {
    ypdr200_x0b_resp_param_hdr_t hdr;
    uint8_t* pMask; //MSB first
} ypdr200_x0b_resp_param_t;

void ypdr200_x0b_resp_param_dispose(ypdr200_x0b_resp_param_t* pRespParam);

#define YPDR200_X0B_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_X0B_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_X0B_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define YPDR200_X0B_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
/**
 * @brief Get Select parameter
 * @warning You need to call `ypdr200_x0b_resp_param_dispose` to free the memory allocated for `pRespParam_out` after you won't use it anymore
 */
int ypdr200_x0b(uhfman_ctx_t* pCtx, ypdr200_x0b_resp_param_t* pRespParam_out, ypdr200_resp_err_code_t* pRespErrCode);

#endif // YPDR200_H
