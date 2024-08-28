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
#define YPDR200_X0C_REQ_PARAM_HDR_SIZE YPDR200_X0B_RESP_PARAM_HDR_SIZE
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
} __attribute__((__packed__)) ypdr200_select_param_hdr_t;

typedef ypdr200_select_param_hdr_t ypdr200_x0b_resp_param_hdr_t;

typedef struct {
    ypdr200_x0b_resp_param_hdr_t hdr;
    uint8_t* pMask; //MSB first
} ypdr200_select_param_t;

typedef ypdr200_select_param_t ypdr200_x0b_resp_param_t;

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

//#define YPDR200_X0D_RESP_PARAM_SIZE 2
typedef union {
    struct {
        #if defined(TARGET_LITTLE_ENDIAN)
        uint8_t reserved : 3;
        uint8_t q : 4;
        uint8_t target : 1;
        uint8_t session : 2;
        uint8_t sel : 2;
        uint8_t trext : 1;
        uint8_t m : 2;
        uint8_t dr : 1;
        #elif defined(TARGET_BIG_ENDIAN)
        uint8_t dr : 1;
        uint8_t m : 2;
        uint8_t trext : 1;
        uint8_t sel : 2;
        uint8_t session : 2;
        uint8_t target : 1;
        uint8_t q : 4;
        uint8_t reserved : 3;
        #else
        #error "Neither TARGET_LITTLE_ENDIAN nor TARGET_BIG_ENDIAN is defined"
        #endif
    };
    //uint8_t raw[YPDR200_X0D_RESP_PARAM_SIZE];
    uint16_t raw;
} __attribute__((__packed__)) ypdr200_query_param_t;

typedef ypdr200_query_param_t ypdr200_x0d_resp_param_t;

#define YPDR200_X0D_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_X0D_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_X0D_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define YPDR200_X0D_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
/**
 * @brief Get Query Parameters
 */
int ypdr200_x0d(uhfman_ctx_t* pCtx, ypdr200_x0d_resp_param_t* pRespParam_out, ypdr200_resp_err_code_t* pRespErrCode);

#define YPDR200_XAA_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_XAA_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_XAA_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define YPDR200_XAA_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
/**
 * @brief Get working channel
 */
int ypdr200_xaa(uhfman_ctx_t* pCtx, uint8_t* pChIndex_out, ypdr200_resp_err_code_t* pRespErrCode);

typedef enum {
    YPDR200_REGION_CHINA_9 = 0x01,
    YPDR200_REGION_CHINA_8 = 0x04,
    YPDR200_REGION_US = 0x08,
    YPDR200_REGION_EU = 0x03,
    YPDR200_REGION_KOREA = 0x06
} ypdr200_x08_region_t;

#define YPDR200_X08_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_X08_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_X08_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define YPDR200_X08_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
/**
 * @brief Get work area
 */
int ypdr200_x08(uhfman_ctx_t* pCtx, ypdr200_x08_region_t* pRegion_out, ypdr200_resp_err_code_t* pRespErrCode);

#define YPDR200_XB7_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_XB7_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_XB7_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define YPDR200_XB7_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
/**
 * @brief Acquire transmit power
   @note Example provided by YPD-R200 technical reference: 2000 coresponds to 20dBm
 */
int ypdr200_xb7(uhfman_ctx_t* pCtx, uint16_t* pTxPower_out, ypdr200_resp_err_code_t* pRespErrCode);

#define YPDR200_XB6_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_XB6_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_XB6_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define YPDR200_XB6_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
/**
 * @brief Set transmit power
 * @note Example provided by YPD-R200 technical reference: 2000 coresponds to 20dBm
 */
int ypdr200_xb6(uhfman_ctx_t* pCtx, uint16_t txPower, ypdr200_resp_err_code_t* pRespErrCode);

typedef enum {
    YPDR200_XF1_RX_DEMOD_MIXER_G_0DB = 0x00,
    YPDR200_XF1_RX_DEMOD_MIXER_G_3DB = 0x01,
    YPDR200_XF1_RX_DEMOD_MIXER_G_6DB = 0x02,
    YPDR200_XF1_RX_DEMOD_MIXER_G_9DB = 0x03,
    YPDR200_XF1_RX_DEMOD_MIXER_G_12DB = 0x04,
    YPDR200_XF1_RX_DEMOD_MIXER_G_15DB = 0x05,
   YPDR200_XF1_RX_DEMOD_MIXER_G_16DB = 0x06 
} ypdr200_xf1_rx_demod_mixer_g_t;

typedef enum {
    YPDR200_XF1_RX_DEMOD_IF_G_12DB = 0x00,
    YPDR200_XF1_RX_DEMOD_IF_G_18DB = 0x01,
    YPDR200_XF1_RX_DEMOD_IF_G_21DB = 0x02,
    YPDR200_XF1_RX_DEMOD_IF_G_24DB = 0x03,
    YPDR200_XF1_RX_DEMOD_IF_G_27DB = 0x04,
    YPDR200_XF1_RX_DEMOD_IF_G_30DB = 0x05,
    YPDR200_XF1_RX_DEMOD_IF_G_36DB = 0x06,
    YPDR200_XF1_RX_DEMOD_IF_G_40DB = 0x07
} ypdr200_xf1_rx_demod_if_g_t;

//#define YPDR200_XF1_DEMOD_PARAMS_SIZE 4
typedef union {
    struct {
#if defined(TARGET_LITTLE_ENDIAN)
        uint8_t thrdLsb;
        uint8_t thrdMsb;
        ypdr200_xf1_rx_demod_if_g_t if_G : 8;
        ypdr200_xf1_rx_demod_mixer_g_t mixer_G : 8;
#elif defined(TARGET_BIG_ENDIAN)
        ypdr200_xf1_rx_demod_mixer_g_t mixer_G : 8;
        ypdr200_xf1_rx_demod_if_g_t if_G : 8;
        uint8_t thrdMsb;
        uint8_t thrdLsb;
#else
#error "Neither TARGET_LITTLE_ENDIAN nor TARGET_BIG_ENDIAN is defined"
#endif
    };
    uint32_t raw;
} __attribute__((__packed__)) ypdr200_xf1_rx_demod_params_t;

#define YPDR200_XF1_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_XF1_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_XF1_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define YPDR200_XF1_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
/**
 * @brief Get the parameters of receiver demodulator
 */
int ypdr200_xf1(uhfman_ctx_t* pCtx, ypdr200_xf1_rx_demod_params_t* pParams_out, ypdr200_resp_err_code_t* pRespErrCode);

#define YPDR200_X22_NTF_PARAM_SIZE 17
#define YPDR200_X22_NTF_PARAM_EPC_LENGTH 12
typedef union {
    struct {
        uint8_t rssi;
        uint8_t pc[2];
        uint8_t epc[YPDR200_X22_NTF_PARAM_EPC_LENGTH]; // TODO #ae3759b4 variabilize using tag memory read and write commands provided by YPD-R200's M100 chip
        uint8_t crc[2];
    };
    uint8_t raw[YPDR200_X22_NTF_PARAM_SIZE]; // TODO support longer EPC?
} __attribute__((__packed__)) ypdr200_x22_ntf_param_t;

typedef void (*ypdr200_x22_callback)(ypdr200_x22_ntf_param_t ntfParam, const void* pUserData);

#define YPDR200_X22_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_X22_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_X22_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define YPDR200_X22_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define YPDR200_X22_ERR_READ_NOTIFICATION UHFMAN_ERR_READ_NOTIFICATION
#define YPDR200_X22_ERR_UNEXPECTED_FRAME_TYPE UHFMAN_ERR_UNEXPECTED_FRAME_TYPE
/**
 * @brief Single polling instruction
 */
int ypdr200_x22(uhfman_ctx_t* pCtx,  ypdr200_resp_err_code_t* pRespErrCode, ypdr200_x22_callback cb, const void* pCbUserData);

#define YPDR200_X27_REQ_PARAM_SIZE 3
typedef union {
    struct {
        uint8_t reserved;
        uint8_t cntMsb;
        uint8_t cntLsb;
    };
    uint8_t raw[YPDR200_X27_REQ_PARAM_SIZE];
} __attribute__((__packed__)) ypdr200_x27_req_param_t;

ypdr200_x27_req_param_t ypdr200_x27_req_param_make(uint16_t cnt);

#define YPDR200_X27_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_X27_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_X27_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define YPDR200_X27_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define YPDR200_X27_ERR_READ_NOTIFICATION UHFMAN_ERR_READ_NOTIFICATION
#define YPDR200_X27_ERR_UNEXPECTED_FRAME_TYPE UHFMAN_ERR_UNEXPECTED_FRAME_TYPE
/**
 * @brief Multiple polling instruction
 */
int ypdr200_x27(uhfman_ctx_t* pCtx, ypdr200_x27_req_param_t param, ypdr200_resp_err_code_t* pRespErrCode, ypdr200_x22_callback cb, const void* pCbUserData);

#define YPDR200_X28_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_X28_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_X28_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define YPDR200_X28_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
/**
 * @brief Stop multiple polling instructions
*/
int ypdr200_x28(uhfman_ctx_t* pCtx, ypdr200_resp_err_code_t* pRespErrCode);

#define YPDR200_X12_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_X12_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_X12_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define YPDR200_X12_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
/**
 * @brief Set send select instruction mode
 */
int ypdr200_x12(uhfman_ctx_t* pCtx, uint8_t mode, ypdr200_resp_err_code_t* pRespErrCode);

typedef ypdr200_select_param_hdr_t ypdr200_x0c_req_param_hdr_t;
typedef ypdr200_select_param_t ypdr200_x0c_req_param_t;

ypdr200_x0c_req_param_t ypdr200_x0c_req_param_make(ypdr200_x0c_req_param_hdr_t hdr, uint8_t* pMask);

void ypdr200_x0c_req_param_dispose(ypdr200_x0c_req_param_t* pReqParam);

#define YPDR200_X0C_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_X0C_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_X0C_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define YPDR200_X0C_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
/**
 * @brief Set select parameter instruction
 */
int ypdr200_x0c(uhfman_ctx_t* pCtx, ypdr200_x0c_req_param_t* pReqParam, ypdr200_resp_err_code_t* pRespErrCode);

typedef ypdr200_query_param_t ypdr200_x0e_req_param_t;

#define YPDR200_X0E_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_X0E_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_X0E_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define YPDR200_X0E_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
/**
 * @brief Set Query parameters
 */
int ypdr200_x0e(uhfman_ctx_t* pCtx, ypdr200_x0e_req_param_t param, ypdr200_resp_err_code_t* pRespErrCode);

typedef union {
    struct {
#if defined(TARGET_LITTLE_ENDIAN)
        uint8_t ch_h;
        uint8_t ch_l;
#elif defined(TARGET_BIG_ENDIAN)
        uint8_t ch_l;
        uint8_t ch_h;
#else
#error "Neither TARGET_LITTLE_ENDIAN nor TARGET_BIG_ENDIAN is defined"
#endif
    };
    uint16_t raw;
} __attribute__((__packed__)) ypdr200_jmdr_hdr_t;

typedef struct { // Jamming detection result
    ypdr200_jmdr_hdr_t hdr;
    uint8_t* pJmr; // Contains respective RSSI values for each channel (the tech ref says it's MSB first - it probably means the first byte is the RSSI value for the channel with the highest index - TODO verify this)
} ypdr200_jmdr_t;

void ypdr200_jmdr_dispose(ypdr200_jmdr_t* pJmdr);

typedef ypdr200_jmdr_hdr_t ypdr200_xf3_resp_param_hdr_t;
typedef ypdr200_jmdr_t ypdr200_xf3_resp_param_t;

/**
 * @brief Get the length of the JMR array (number of channels)
 */
uint16_t ypdr200_jmdr_get_jmr_len(ypdr200_xf3_resp_param_t* pRespParam);

#define YPDR200_XF3_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_XF3_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_XF3_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define YPDR200_XF3_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
/**
 * @brief Test channel RSSI
 * @note You need to call `ypdr200_jmdr_dispose` to free the memory allocated for `pRespParam_out` after you won't use it anymore
 */
int ypdr200_xf3(uhfman_ctx_t* pCtx, ypdr200_xf3_resp_param_t* pRespParam_out, ypdr200_resp_err_code_t* pRespErrCode);

typedef ypdr200_jmdr_hdr_t ypdr200_xf2_resp_param_hdr_t;
typedef ypdr200_jmdr_t ypdr200_xf2_resp_param_t;

#define YPDR200_XF2_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_XF2_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_XF2_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define YPDR200_XF2_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
/**
 * @brief Test RF input blocking signal (channelwise jamming detection as far as I understand)
 */
int ypdr200_xf2(uhfman_ctx_t* pCtx, ypdr200_xf2_resp_param_t* pRespParam_out, ypdr200_resp_err_code_t* pRespErrCode);

// TODO define data structures for the following commands
// TODO buy 50pcs small tags together with the missing ones

#define YPDR200_X39_REQ_PARAM_SIZE 9
typedef union {
    struct {
        uint8_t ap[4];
        uint8_t memBank;
        uint8_t sa[2];
        uint8_t dl[2];
    };
    uint8_t raw[YPDR200_X39_REQ_PARAM_SIZE];
} __attribute__((__packed__)) ypdr200_x39_req_param_t;

#define YPDR200_X39_RESP_PARAM_HDR_SIZE 15
typedef union {
    struct {
        uint8_t ul;
        uint8_t pc[2];
        uint8_t epc[12];
    };
    uint8_t raw[YPDR200_X39_RESP_PARAM_HDR_SIZE]; // TODO support longer EPC by conforming to the `ul` value?
} __attribute__((__packed__)) ypdr200_x39_resp_param_hdr_t;

typedef struct {
    uint16_t dataLen;
    uint8_t* pData;
} ypdr200_x39_resp_param_body_t;

typedef struct {
    ypdr200_x39_resp_param_hdr_t hdr;
    ypdr200_x39_resp_param_body_t body;
} ypdr200_x39_resp_param_t;

void ypdr200_x39_resp_param_dispose(ypdr200_x39_resp_param_t* pRespParam);

#define YPDR200_X39_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_X39_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_X39_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define YPDR200_X39_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
/**
 * @brief Read label data storage area
 */
int ypdr200_x39(uhfman_ctx_t* pCtx, ypdr200_x39_req_param_t param, ypdr200_x39_resp_param_t* pRespParam_out, ypdr200_resp_err_code_t* pRespErrCode);

#define YPDR200_X49_REQ_PARAM_HDR_SIZE 9
typedef union {
    struct {
        uint8_t ap[4];
        uint8_t memBank;
        uint8_t sa[2];
        uint8_t dl[2];
    };
    uint8_t raw[YPDR200_X49_REQ_PARAM_HDR_SIZE];
} __attribute__((__packed__)) ypdr200_x49_req_param_hdr_t;

typedef struct {
    ypdr200_x49_req_param_hdr_t hdr;
    uint8_t* pDt;
} ypdr200_x49_req_param_t;

/**
 * @note ap needs to be supplied MSB first
 */
ypdr200_x49_req_param_hdr_t ypdr200_x49_req_param_hdr_make(uint16_t sa, uint16_t dl, uint8_t memBank, uint8_t ap[4]);

ypdr200_x49_req_param_t ypdr200_x49_req_param_make(ypdr200_x49_req_param_hdr_t hdr, uint8_t* pDt);

void ypdr200_x49_req_param_dispose(ypdr200_x49_req_param_t* pReqParam);

#define YPDR200_X49_RESP_PARAM_SIZE 16
typedef union {
    struct {
        uint8_t ul;
        uint8_t pc[2];
        uint8_t epc[12];
        uint8_t parameter;
    };
    uint8_t raw[YPDR200_X49_RESP_PARAM_SIZE]; // TODO support longer EPC by conforming to the `ul` value?
} __attribute__((__packed__)) ypdr200_x49_resp_param_t;

#define YPDR200_X49_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_X49_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define YPDR200_X49_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define YPDR200_X49_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
/**
 * @brief Write label data storage area
 */
int ypdr200_x49(uhfman_ctx_t* pCtx, ypdr200_x49_req_param_t param, ypdr200_x49_resp_param_t* pRespParam_out, ypdr200_resp_err_code_t* pRespErrCode);

#endif // YPDR200_H
