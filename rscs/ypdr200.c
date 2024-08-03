#include "ypdr200.h"
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
//#include <stdio.h>
//#include <fcntl.h>
#include <unistd.h>
#include <termios.h>

#include "log.h"

// lsusb -d1a86:7523 -v

#define YPDR200_BULK_ENDPOINT_ADDR_IN 0x82U
#define YPDR200_BULK_ENDPOINT_ADDR_OUT 0x02U
#define YPDR200_INTERRUPT_ENDPOINT_ADDR_IN 0x81U

#define YPDR200_BULK_TRANSFER_TIMEOUT_MS 5000

#define YPDR200_FRAME_PROLOG_SIZE 5
typedef union _ypdr200_frame_prolog {
    struct {
        uint8_t header;
        uint8_t type;
        uint8_t cmd;
        uint8_t paramLengthMsb;
        uint8_t paramLengthLsb;
    };
    uint8_t raw[YPDR200_FRAME_PROLOG_SIZE];
} __attribute__((__packed__)) ypdr200_frame_prolog_t;

#define YPDR200_FRAME_EPILOG_SIZE 2
typedef union _ypdr200_frame_epilog {
    struct {
        uint8_t checksum;
        uint8_t end;
    };
    uint8_t raw[YPDR200_FRAME_EPILOG_SIZE];
} __attribute__((__packed__)) ypdr200_frame_epilog_t;

typedef struct _ypdr200_frame {
    ypdr200_frame_prolog_t prolog;
    ypdr200_frame_epilog_t epilog;
    uint8_t* pParamData;
} ypdr200_frame_t;

/**
 * @brief Frees pParamData of the provided frame, thus preventing a memory leak after pFrame is not used anymore
 * @warning Only use this function if pParamData was allocated at runtime
 */
void ypdr200_frame_dispose(ypdr200_frame_t* pFrame) {
    if (pFrame->pParamData != NULL) {
        free (pFrame->pParamData);
    }
}

static uint16_t ypdr200_frame_prolog_get_param_length(const ypdr200_frame_prolog_t* pProlog) {
    uint16_t paramLen = (uint16_t)(pProlog->paramLengthMsb);
    paramLen = (paramLen << 8) | (uint16_t)(pProlog->paramLengthLsb);
    return paramLen;
}

static uint8_t ypdr200_frame_compute_checksum(const ypdr200_frame_t* pFrame) {
    uint32_t sum = 0;
    ypdr200_frame_prolog_t prolog = pFrame->prolog;
    sum += prolog.type;
    sum += prolog.cmd;
    sum += prolog.paramLengthMsb;
    sum += prolog.paramLengthLsb;
    uint16_t paramLen = ypdr200_frame_prolog_get_param_length(&pFrame->prolog);
    if (paramLen > 0) {
        assert(pFrame->pParamData != (uint8_t*)0);
    }
    for (uint16_t i = 0; i < paramLen; i++) {
        sum += pFrame->pParamData[i];
    }
    return (uint8_t)(sum & 0xFF);
}

static uint32_t ypdr200_frame_length(const ypdr200_frame_t* pFrame) {
    uint16_t paramLen = ypdr200_frame_prolog_get_param_length(&pFrame->prolog);
    return (uint32_t)(YPDR200_FRAME_PROLOG_SIZE) + (uint32_t)(paramLen) + (uint32_t)(YPDR200_FRAME_EPILOG_SIZE);
}

/**
 * @brief Converts a frame to a raw buffer ready to be sent to the device
 * @note You need to free the output buffer after use
 * @param pFrame Pointer to the frame structure to be serialized
 */
static uint8_t* ypdr200_frame_raw(const ypdr200_frame_t* pFrame) {
    if (pFrame == (const ypdr200_frame_t*)0) {
        errno = EINVAL;
        return (uint8_t*)0;
    }
    ypdr200_frame_prolog_t prolog = pFrame->prolog;
    uint16_t paramLen = ypdr200_frame_prolog_get_param_length(&prolog);    
    uint8_t* pRaw = (uint8_t*) malloc(YPDR200_FRAME_PROLOG_SIZE + paramLen + YPDR200_FRAME_EPILOG_SIZE);
    if (pRaw == (uint8_t*)0) {
        errno = ENOMEM;
        return (uint8_t*)0;
    }
    uint8_t* pBlk = pRaw;
    memcpy(pBlk, prolog.raw, sizeof(prolog.raw));
    pBlk += sizeof(prolog.raw);
    if (paramLen > 0) {
        assert(pFrame->pParamData != (uint8_t*)0);
        memcpy(pBlk, pFrame->pParamData, (size_t)paramLen);
        pBlk += (size_t)paramLen;
    }
    memcpy(pBlk, pFrame->epilog.raw, sizeof(pFrame->epilog.raw));
    return pRaw;
}

typedef enum ypdr200_frame_type {
    YPDR200_FRAME_TYPE_COMMAND = 0x00,
    YPDR200_FRAME_TYPE_RESPONSE = 0x01,
    YPDR200_FRAME_TYPE_NOTIFICATION = 0x02
} ypdr200_frame_type_t;

typedef enum ypdr200_frame_cmd {
    YPDR200_FRAME_CMD_X03 = 0x03,
    YPDR200_FRAME_CMD_X08 = 0x08,
    YPDR200_FRAME_CMD_X0B = 0x0b,
    YPDR200_FRAME_CMD_X0D = 0x0d,
    YPDR200_FRAME_CMD_X11 = 0x11,
    YPDR200_FRAME_CMD_X22 = 0x22,
    YPDR200_FRAME_CMD_X27 = 0x27,
    YPDR200_FRAME_CMD_XAA = 0xaa,
    YPDR200_FRAME_CMD_XB7 = 0xb7,
    YPDR200_FRAME_CMD_XF1 = 0xf1,
    YPDR200_FRAME_CMD_XFF = 0xff
} ypdr200_frame_cmd_t;

/**
 * @note The buffer pointed by pParamData will be directly referenced by the frame structure
 */
static void ypdr200_frame_set_param_data(ypdr200_frame_t* pFrame, uint8_t* pParamData) {
    pFrame->pParamData = pParamData;
}

static void ypdr200_frame_set_checksum(ypdr200_frame_t* pFrame) {
    pFrame->epilog.checksum = ypdr200_frame_compute_checksum(pFrame);
}

/**
 * @note The buffer pointed by pParamData will be directly referenced by the frame structure
 */
static ypdr200_frame_t ypdr200_frame_construct(ypdr200_frame_type_t type, ypdr200_frame_cmd_t cmd, uint16_t paramLen, uint8_t* pParamData) {
    ypdr200_frame_t frame = (ypdr200_frame_t) {
        .prolog = (ypdr200_frame_prolog_t) {
            .header = 0xAA,
            .type = (uint8_t)type,
            .cmd = (uint8_t)cmd,
            .paramLengthMsb = (uint8_t)(paramLen >> 8),
            .paramLengthLsb = (uint8_t)(paramLen & 0xFF)
        },
        .epilog = (ypdr200_frame_epilog_t) {
            .checksum = 0x00,
            .end = 0xDD
        },
        .pParamData = (uint8_t*)0
    };

    ypdr200_frame_set_param_data(&frame, pParamData);
    ypdr200_frame_set_checksum(&frame);
    return frame;
}

#define YPDR200_FRAME_RECV_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_FRAME_RECV_ERR_READ UHFMAN_ERR_READ_RESPONSE
#define YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME UHFMAN_ERR_ERROR_RESPONSE
#define YPDR200_FRAME_RECV_ERR_CKSUM_MISMATCH (UHFMAN_ERR_OTHER + 0)
/**
 * @note You need to call `ypdr200_frame_dispose` on `pFrameRcv` after it is no longer used 
 */
static int ypdr200_frame_recv(ypdr200_frame_t* pFrameRcv, uhfman_ctx_t* pCtx, ypdr200_frame_cmd_t expectedCmd, ypdr200_resp_err_code_t* pRcvErr) {
    if (pFrameRcv == NULL) { assert(0); }
    if (pCtx == NULL) { assert(0); }

    ypdr200_frame_prolog_t prologIn = {};
    int actual_size_received = 0;

    #if YPDR200_INTERFACE_TYPE == YPDR200_INTERFACE_TYPE_LIBUSB
    rv = libusb_bulk_transfer(pCtx->handle, YPDR200_BULK_ENDPOINT_ADDR_IN, prologIn.raw, YPDR200_FRAME_PROLOG_SIZE, &actual_size_received, YPDR200_BULK_TRANSFER_TIMEOUT_MS);
    if (rv != 0) {
        LOG_E("Error reading response prolog: %s (%d)", libusb_error_name(rv), rv);
        return YPDR200_FRAME_RECV_ERR_READ;
    }
    #elif YPDR200_INTERFACE_TYPE == YPDR200_INTERFACE_TYPE_SERIAL
    actual_size_received = read(pCtx->fd, prologIn.raw, YPDR200_FRAME_PROLOG_SIZE);
    #endif
    if (actual_size_received != YPDR200_FRAME_PROLOG_SIZE) {
        LOG_E("Error reading response prolog: YPD200_FRAME_PROLOG_SIZE=%d, actual_size_received=%d", YPDR200_FRAME_PROLOG_SIZE, actual_size_received);
        LOG_E_TBC("The received incomplete prolog is: ");
        for (int i = 0; i < actual_size_received; i++) {
            LOG_E_CTBC("0x%02X ", prologIn.raw[i]);
        }
        LOG_E_CFIN("");
        return YPDR200_FRAME_RECV_ERR_READ;
    }

    ///<debug print received prolog>
    LOG_V_TBC("Received prolog: ");
    for (int i = 0; i < YPDR200_FRAME_PROLOG_SIZE; i++) {
        LOG_V_CTBC("0x%02X ", prologIn.raw[i]);
    }
    LOG_V_CFIN("");

    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&prologIn);
    uint8_t* pParamIn = (uint8_t*)0;
    if (paramInLen > 0) {
        pParamIn = (uint8_t*) malloc(paramInLen);
        if (pParamIn == (uint8_t*)0) {
            errno = ENOMEM;
            return YPDR200_FRAME_RECV_ERR_READ;
        }
        actual_size_received = 0;
        #if YPDR200_INTERFACE_TYPE == YPDR200_INTERFACE_TYPE_LIBUSB
        rv = libusb_bulk_transfer(pCtx->handle, YPDR200_BULK_ENDPOINT_ADDR_IN, pParamIn, paramInLen, &actual_size_received, YPDR200_BULK_TRANSFER_TIMEOUT_MS);
        if (rv != 0) {
            LOG_E("Error reading param data: %s (%d)", libusb_error_name(rv), rv);
            free(pParamIn);
            return YPDR200_FRAME_RECV_ERR_READ;
        }
        #elif YPDR200_INTERFACE_TYPE == YPDR200_INTERFACE_TYPE_SERIAL
        while (actual_size_received < paramInLen) {
            ssize_t nread = read(pCtx->fd, pParamIn, paramInLen - actual_size_received);
            if (nread <= 0) {
                LOG_W("Error reading param data: paramInLen=%d, actual_size_received=%d, nread=%d", paramInLen, actual_size_received, nread);
                free(pParamIn);
                return YPDR200_FRAME_RECV_ERR_READ;
            }
            actual_size_received += nread;
        }
        #endif
        // if (actual_size_received != paramInLen) {
        //     fprintf(stderr, "Error reading param data: paramInLen=%d, actual_size_received=%d\n", paramInLen, actual_size_received);
        //     free(pParamIn);
        //     return YPDR200_FRAME_RECV_ERR_READ;
        // }

        ///<debug print received param data>
        LOG_V_TBC("Received param data: ");
        for (int i = 0; i < paramInLen; i++) {
            LOG_V_CTBC("0x%02X ", pParamIn[i]);
        }
        LOG_V_CFIN("");
        ///</debug print received param data>
    }

    ypdr200_frame_epilog_t epilogIn = {};
    actual_size_received = 0;
    #if YPDR200_INTERFACE_TYPE == YPDR200_INTERFACE_TYPE_LIBUSB
    rv = libusb_bulk_transfer(pCtx->handle, YPDR200_BULK_ENDPOINT_ADDR_IN, epilogIn.raw, YPDR200_FRAME_EPILOG_SIZE, &actual_size_received, YPDR200_BULK_TRANSFER_TIMEOUT_MS);
    if (rv != 0) {
        LOG_E("Error reading epilog: %s (%d)", libusb_error_name(rv), rv);
        if (pParamIn != (uint8_t*)0) {
            free(pParamIn);
        }
        return YPDR200_FRAME_RECV_ERR_READ;
    }
    #elif YPDR200_INTERFACE_TYPE == YPDR200_INTERFACE_TYPE_SERIAL
    actual_size_received = read(pCtx->fd, epilogIn.raw, YPDR200_FRAME_EPILOG_SIZE);
    #endif
    if (actual_size_received != YPDR200_FRAME_EPILOG_SIZE) {
        LOG_E("Error reading epilog: YPD200_FRAME_EPILOG_SIZE=%d, actual_size_received=%d", YPDR200_FRAME_EPILOG_SIZE, actual_size_received);
        if (pParamIn != (uint8_t*)0) {
            free(pParamIn);
        }
        return YPDR200_FRAME_RECV_ERR_READ;
    }

    ///<debug print received epilog>
    LOG_V_TBC("Received epilog: ");
    for (int i = 0; i < YPDR200_FRAME_EPILOG_SIZE; i++) {
        LOG_V_CTBC("0x%02X ", epilogIn.raw[i]);
    }
    LOG_V_CFIN("");
    ///</debug print received epilog>

    ypdr200_frame_t frameIn = (ypdr200_frame_t) {
        .prolog = prologIn,
        .epilog = epilogIn,
        .pParamData = pParamIn
    };

    /// <Print the received frame for debugging>
    LOG_D_TBC("Received frame: ");
    uint8_t *pRawIn = ypdr200_frame_raw(&frameIn);
    uint32_t frameInLen = ypdr200_frame_length(&frameIn);
    for (uint32_t i = 0; i < frameInLen; i++) {
        LOG_D_CTBC("0x%02X ", pRawIn[i]);
    }
    LOG_D_CFIN("");
    free(pRawIn);
    /// </Print the received frame for debugging>

    uint8_t checksum = ypdr200_frame_compute_checksum(&frameIn);
    if (checksum != epilogIn.checksum) {
        LOG_W("Checksum mismatch: expected=0x%02X, actual=0x%02X", checksum, epilogIn.checksum); // TODO don't ignore
        if (pParamIn != (uint8_t*)0) {
            free(pParamIn);
        }
        return YPDR200_FRAME_RECV_ERR_CKSUM_MISMATCH;
        //return YPDR200_FRAME_RECV_ERR_READ;
        //LOG_I("Ignoring checksum mismatch!\n"); // TODO don't ignore
    } else {
        LOG_V("Checksum OK");
    }

    if (frameIn.prolog.type != YPDR200_FRAME_TYPE_RESPONSE && frameIn.prolog.type != YPDR200_FRAME_TYPE_NOTIFICATION) {
        LOG_W("Unexpected frame type: expected=0x%02X or 0x%02X, actual=0x%02X", YPDR200_FRAME_TYPE_RESPONSE, YPDR200_FRAME_TYPE_NOTIFICATION, frameIn.prolog.type);
        if (pParamIn != (uint8_t*)0) {
            free(pParamIn);
        }
        return YPDR200_FRAME_RECV_ERR_READ;
    }

    if (! (expectedCmd == YPDR200_FRAME_CMD_X27 && frameIn.prolog.cmd == YPDR200_FRAME_CMD_X22)) {
        if (frameIn.prolog.cmd != (uint8_t)expectedCmd) {
            if (frameIn.prolog.cmd == (uint8_t)YPDR200_FRAME_CMD_XFF) {
                uint16_t respParamLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
                if (respParamLen != 1) {
                    if (pParamIn != (uint8_t*)0) {
                        free(pParamIn);
                    }
                    LOG_W("Received error response frame, but resp param had unexpected length %u\n", respParamLen);
                    return YPDR200_FRAME_RECV_ERR_READ;
                }
                if (pParamIn == NULL) {
                    LOG_W("Received error response frame, but resp param is NULL (unlikely)\n");
                    return YPDR200_FRAME_RECV_ERR_READ;
                }
                LOG_D("Received error response frame with error code 0x%02X", pParamIn[0]);
                *pRcvErr = (ypdr200_resp_err_code_t)(pParamIn[0]);
                if (pParamIn != (uint8_t*)0) {
                    free(pParamIn);
                }

                return YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME;
            }
            LOG_W("Unexpected frame cmd: expected=0x%02X, actual=0x%02X", (uint8_t)expectedCmd, frameIn.prolog.cmd);
            if (pParamIn != (uint8_t*)0) {
                free(pParamIn);
            }
            return YPDR200_FRAME_RECV_ERR_READ;
        }
    }

    *pFrameRcv = frameIn;
    return YPDR200_FRAME_RECV_ERR_SUCCESS;
}

#define YPDR200_FRAME_SEND_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define YPDR200_FRAME_SEND_ERR_SEND UHFMAN_ERR_SEND_COMMAND
static int ypdr200_frame_send(ypdr200_frame_t* pFrameSnd, uhfman_ctx_t* pCtx) {
    if (pFrameSnd == NULL) { assert(0); }
    if (pCtx == NULL) { assert(0); }

    uint8_t* pDataOut = ypdr200_frame_raw(pFrameSnd);
    uint32_t dataOutLen = ypdr200_frame_length(pFrameSnd);

    ///<debug print raw data for sending>
    LOG_V_TBC("Sending frame: ");
    for (uint32_t i = 0; i < dataOutLen; i++) {
        LOG_V_CTBC("0x%02X ", pDataOut[i]);
    }
    LOG_V_CFIN("");
    ///</debug print raw data for sending>

    int actual_size_transmitted = 0;
#if YPDR200_INTERFACE_TYPE == YPDR200_INTERFACE_TYPE_LIBUSB
    int rv = libusb_bulk_transfer(pCtx->handle, YPDR200_BULK_ENDPOINT_ADDR_OUT, pDataOut, dataOutLen, &actual_size_transmitted, YPDR200_BULK_TRANSFER_TIMEOUT_MS);
    if (rv != 0) {
        LOG_E("Error sending cmd 0x%02X: %s (%d)", pFrameSnd->prolog.cmd, libusb_error_name(rv), rv);
        return YPDR200_FRAME_SEND_ERR_SEND;
    }
#elif YPDR200_INTERFACE_TYPE == YPDR200_INTERFACE_TYPE_SERIAL
    actual_size_transmitted = write(pCtx->fd, pDataOut, dataOutLen);
#else
    #error "Unsupported device interface"
#endif
    if ((uint32_t)actual_size_transmitted != dataOutLen) {
        LOG_E("Error sending cmd 0x%02X: dataOutLen=%d, actual_size_transmitted=%d", pFrameSnd->prolog.cmd, dataOutLen, actual_size_transmitted);
        free(pDataOut);
        return YPDR200_FRAME_SEND_ERR_SEND;
    }
    free(pDataOut);
    return YPDR200_FRAME_SEND_ERR_SUCCESS;
}

// Example frame raw data: AA 00 11 00 02 04 80 97 DD
int ypdr200_x11(uhfman_ctx_t* pCtx, ypdr200_x11_param_t baudRate) {
    uint16_t pow = (uint16_t)(((uint32_t)baudRate) / 100);
    uint8_t powMsb = (uint8_t)(pow >> 8);
    uint8_t powLsb = (uint8_t)(pow & 0xFF);
    uint8_t paramData[2] = {powMsb, powLsb};
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_X11, 2U, paramData);

    int err = ypdr200_frame_send(&frameOut, pCtx);
    // No need to call ypdr200_frame_dispose, as paramData is allocated statically

    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_X11_ERR_SEND_COMMAND;
    }

    // No response expected
    return YPDR200_X11_ERR_SUCCESS;
}

// Example frame raw data: AA 00 03 00 01 infoType 04+infoType DD
int ypdr200_x03(uhfman_ctx_t* pCtx, ypdr200_x03_req_param_t infoType, char** ppcInfo_out, ypdr200_resp_err_code_t* pRespErrCode) {
    uint8_t param = (uint8_t)infoType;
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_X03, 1U, &param);

    int err = ypdr200_frame_send(&frameOut, pCtx);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_X03_ERR_SEND_COMMAND;
    }

    ypdr200_frame_t frameIn = {};
    err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_X03, pRespErrCode);
    if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) {
        if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
            return YPDR200_X03_ERR_ERROR_RESPONSE;
        }
        return YPDR200_X03_ERR_READ_RESPONSE;
    }

    ypdr200_frame_prolog_t prologIn = frameIn.prolog;
    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&prologIn);
    uint8_t* pParamIn = frameIn.pParamData;

    if (pParamIn) {
        if (paramInLen < 1) {
            LOG_W("Unexpected param data length: expected>0, actual=%d", paramInLen);
            //free(pParamIn);
            ypdr200_frame_dispose(&frameIn);
            return YPDR200_X03_ERR_READ_RESPONSE;
        }
        if (pParamIn[0] != (uint8_t)infoType) {
            LOG_W("Unexpected infoType byte found in received param data: expected=0x%02X, actual=0x%02X", (uint8_t)infoType, pParamIn[0]);
            //free(pParamIn);
            ypdr200_frame_dispose(&frameIn);
            return YPDR200_X03_ERR_READ_RESPONSE;
        }
        *ppcInfo_out = (char*) malloc(paramInLen); // paramInLen-1 + null terminator
        if (*ppcInfo_out == (char*)0) {
            errno = ENOMEM;
            // if (pParamIn != (uint8_t*)0) {
            //     free(pParamIn);
            // }
            ypdr200_frame_dispose(&frameIn);
            return YPDR200_X03_ERR_READ_RESPONSE;
        }
        memcpy(*ppcInfo_out, pParamIn + 1, paramInLen); // the byte of the buffer behind pParamIn is infoType (expected to be the same as infoType provided by the function caller)
        (*ppcInfo_out)[paramInLen-1] = '\0';

        //free(pParamIn);
        ypdr200_frame_dispose(&frameIn);
    } else {
        *ppcInfo_out = (char*) malloc(1);
        if (*ppcInfo_out == (char*)0) {
            errno = ENOMEM;
            return YPDR200_X03_ERR_READ_RESPONSE;
        }
        (*ppcInfo_out)[0] = '\0';
    }

    return YPDR200_X03_ERR_SUCCESS;
}

/**
 * @warning Need to check outside of this function if the pointer to be used is valid
 */
static void ypdr200_x0b_resp_param_set_hdr_from_raw(uint8_t* pRawParamData, ypdr200_x0b_resp_param_t* pRespParam) {
    // return (ypdr200_x0b_resp_param_t) {
    //     .hdr = (ypdr200_x0b_resp_param_hdr_t) {
    //         .raw = pRawParamData
    //     },
    //     .pMask = pRawParamData + YPDR200_X0B_RESP_PARAM_HDR_SIZE
    // };
    if (pRespParam != NULL) {
        ypdr200_x0b_resp_param_t respParam = {};
        memcpy(respParam.hdr.raw, pRawParamData, YPDR200_X0B_RESP_PARAM_HDR_SIZE);
        *pRespParam = respParam;
    }
}

/**
 * @warning Need to check outside of this function if the pointer to be used is valid
 * @returns 0 on success, -1 on error (check errno)
 */
static int ypdr200_x0b_resp_param_set_mask(uint8_t* pMask, ypdr200_x0b_resp_param_t* pRespParam) {
    pRespParam->pMask = malloc(pRespParam->hdr.maskLen);
    if (pRespParam->pMask == (uint8_t*)0) {
        errno = ENOMEM;
        return -1;
    }
    memcpy(pRespParam->pMask, pMask, pRespParam->hdr.maskLen);
    return 0;
}

void ypdr200_x0b_resp_param_dispose(ypdr200_x0b_resp_param_t* pRespParam) {
    if (pRespParam->pMask != (uint8_t*)0) {
        free(pRespParam->pMask);
    }
}

int ypdr200_x0b(uhfman_ctx_t* pCtx, ypdr200_x0b_resp_param_t* pRespParam_out, ypdr200_resp_err_code_t* pRespErrCode) {
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_X0B, 0U, NULL);
    int err = ypdr200_frame_send(&frameOut, pCtx);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_X0B_ERR_SEND_COMMAND;
    }

    ypdr200_frame_t frameIn = {};
    err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_X0B, pRespErrCode);
    if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) {
        if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
            return YPDR200_X0B_ERR_ERROR_RESPONSE;
        }
        return YPDR200_X0B_ERR_READ_RESPONSE;
    }

    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
    if (paramInLen < YPDR200_X0B_RESP_PARAM_HDR_SIZE) {
        int expectedParamInLen = YPDR200_X0B_RESP_PARAM_HDR_SIZE;
        LOG_W("Unexpected param data length: expected>%d, actual=%d", expectedParamInLen, paramInLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X03_ERR_READ_RESPONSE;
    }

    ypdr200_x0b_resp_param_t respParam = {};
    ypdr200_x0b_resp_param_set_hdr_from_raw(frameIn.pParamData, &respParam);
    if (respParam.hdr.maskLen + YPDR200_X0B_RESP_PARAM_HDR_SIZE != paramInLen) {
        LOG_W("Unexpected mask length: expected=%d, actual=%d", respParam.hdr.maskLen, paramInLen - YPDR200_X0B_RESP_PARAM_HDR_SIZE);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X03_ERR_READ_RESPONSE;
    }

    if (-1 == ypdr200_x0b_resp_param_set_mask(frameIn.pParamData + YPDR200_X0B_RESP_PARAM_HDR_SIZE, &respParam)) {
        uhfman_debug_errno(); // print errno
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X03_ERR_READ_RESPONSE;
    }

    ypdr200_frame_dispose(&frameIn);
    
    *pRespParam_out = respParam;
    return YPDR200_X0B_ERR_SUCCESS;
}

int ypdr200_x0d(uhfman_ctx_t* pCtx, ypdr200_x0d_resp_param_t* pRespParam_out, ypdr200_resp_err_code_t* pRespErrCode) {
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_X0D, 0U, NULL);
    int err = ypdr200_frame_send(&frameOut, pCtx);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_X0D_ERR_SEND_COMMAND;
    }

    ypdr200_frame_t frameIn = {};
    err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_X0D, pRespErrCode);
    if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) {
        if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
            return YPDR200_X0D_ERR_ERROR_RESPONSE;
        }
        return YPDR200_X0D_ERR_READ_RESPONSE;
    }

    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
    if (paramInLen != 2) {
        LOG_W("Unexpected param data length: expected=2, actual=%d", paramInLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X0D_ERR_READ_RESPONSE;
    }

    ypdr200_x0d_resp_param_t respParam = {
        .raw = ((uint16_t)(frameIn.pParamData[0]) << 8) | (uint16_t)(frameIn.pParamData[1])
    };

    ypdr200_frame_dispose(&frameIn);

    *pRespParam_out = respParam;
    return YPDR200_X0D_ERR_SUCCESS;
}

int ypdr200_xaa(uhfman_ctx_t* pCtx, uint8_t* pChIndex_out, ypdr200_resp_err_code_t* pRespErrCode) {
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_XAA, 0U, NULL);
    int err = ypdr200_frame_send(&frameOut, pCtx);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_XAA_ERR_SEND_COMMAND;
    }

    ypdr200_frame_t frameIn = {};
    err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_XAA, pRespErrCode);
    if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) {
        if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
            return YPDR200_XAA_ERR_ERROR_RESPONSE;
        }
        return YPDR200_XAA_ERR_READ_RESPONSE;
    }

    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
    if (paramInLen != 1) {
        LOG_W("Unexpected param data length: expected=1, actual=%d", paramInLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_XAA_ERR_READ_RESPONSE;
    }

    *pChIndex_out = frameIn.pParamData[0];

    ypdr200_frame_dispose(&frameIn);
    return YPDR200_XAA_ERR_SUCCESS;
}

int ypdr200_x08(uhfman_ctx_t* pCtx, ypdr200_x08_region_t* pRegion_out, ypdr200_resp_err_code_t* pRespErrCode) {
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_X08, 0U, NULL);
    int err = ypdr200_frame_send(&frameOut, pCtx);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_X08_ERR_SEND_COMMAND;
    }

    ypdr200_frame_t frameIn = {};
    err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_X08, pRespErrCode);
    if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) {
        if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
            return YPDR200_X08_ERR_ERROR_RESPONSE;
        }
        return YPDR200_X08_ERR_READ_RESPONSE;
    }

    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
    if (paramInLen != 1) {
        LOG_W("Unexpected param data length: expected=1, actual=%d", paramInLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X08_ERR_READ_RESPONSE;
    }

    *pRegion_out = (ypdr200_x08_region_t)(frameIn.pParamData[0]);

    ypdr200_frame_dispose(&frameIn);
    return YPDR200_X08_ERR_SUCCESS;
}

int ypdr200_xb7(uhfman_ctx_t* pCtx, uint16_t* pTxPower_out, ypdr200_resp_err_code_t* pRespErrCode) {
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_XB7, 0U, NULL);
    int err = ypdr200_frame_send(&frameOut, pCtx);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_XB7_ERR_SEND_COMMAND;
    }

    ypdr200_frame_t frameIn = {};
    err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_XB7, pRespErrCode);
    if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) {
        if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
            return YPDR200_XB7_ERR_ERROR_RESPONSE;
        }
        return YPDR200_XB7_ERR_READ_RESPONSE;
    }

    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
    if (paramInLen != 2) {
        LOG_W("Unexpected param data length: expected=2, actual=%d", paramInLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_XB7_ERR_READ_RESPONSE;
    }

    *pTxPower_out = ((uint16_t)(frameIn.pParamData[0]) << 8) | (uint16_t)(frameIn.pParamData[1]);

    ypdr200_frame_dispose(&frameIn);
    return YPDR200_XB7_ERR_SUCCESS;
}

int ypdr200_xf1(uhfman_ctx_t* pCtx, ypdr200_xf1_rx_demod_params_t* pParams_out, ypdr200_resp_err_code_t* pRespErrCode) {
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_XF1, 0U, NULL);
    int err = ypdr200_frame_send(&frameOut, pCtx);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_XF1_ERR_SEND_COMMAND;
    }

    ypdr200_frame_t frameIn = {};
    err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_XF1, pRespErrCode);
    if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) {
        if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
            return YPDR200_XF1_ERR_ERROR_RESPONSE;
        }
        return YPDR200_XF1_ERR_READ_RESPONSE;
    }

    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
    if (paramInLen != 4) {
        LOG_W("Unexpected param data length: expected=4, actual=%d", paramInLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_XF1_ERR_READ_RESPONSE;
    }

    assert(sizeof(uint32_t) == 4); // TODO If the code was meant to be run on a platform where sizeof(uint32_t) != 4, the related code would need to be modified to use `raw` as an array of 4 uint8_t elements
    ypdr200_xf1_rx_demod_params_t params = {
        .raw = ((uint32_t)(frameIn.pParamData[0]) << 24) | ((uint32_t)(frameIn.pParamData[1]) << 16) | ((uint32_t)(frameIn.pParamData[2]) << 8) | (uint32_t)(frameIn.pParamData[3])
    };

    ypdr200_frame_dispose(&frameIn);

    *pParams_out = params;
    return YPDR200_XF1_ERR_SUCCESS;
}

#include "utils.h"
/**
 * @returns 1 if the CRC is correct, 0 otherwise
 */
static int ypdr200_x22_ntf_param_check_crc(ypdr200_x22_ntf_param_t* pNtfParam) {
    // size_t u16bufLen = (YPDR200_X22_NTF_PARAM_SIZE - 3) / 2;
    // uint16_t* u16buf = (uint16_t*)malloc(u16bufLen);
    // if (u16buf == NULL) {
    //     errno = ENOMEM;
    //     fprintf(stderr, "Failed to allocate memory for CRC16 check! This system has likely ran out of memory\n");
    //     return 0;
    // }
    // uint16_t crc16 = utils_crc_ccitt_genibus((uint16_t*)(pNtfParam->raw + 1), (YPDR200_X22_NTF_PARAM_SIZE - 3) / 2);
    // utils_buf_u8_to_u16_big_endian(u16buf, (uint8_t*)(pNtfParam->raw + 1), YPDR200_X22_NTF_PARAM_SIZE - 3); // YPD-R200 sends words in big-endian byte order
    // uint16_t crc16 = utils_crc_ccitt_genibus(u16buf, u16bufLen);
   
    //uint16_t crc16 = utils_crc_ccitt_genibus(pNtfParam->raw + 1, YPDR200_X22_NTF_PARAM_SIZE - 3);
    //uint16_t crc16NtfParam = ((uint16_t)(pNtfParam->crc[0]) << 8) | (uint16_t)(pNtfParam->crc[1]);
    //return (int)(crc16 == crc16NtfParam);
    return 1; // TODO Fix CRC16
}

int ypdr200_x22(uhfman_ctx_t* pCtx, ypdr200_resp_err_code_t* pRespErrCode, ypdr200_x22_callback cb, const void* pCbUserData) {
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_X22, 0U, NULL);
    int err = ypdr200_frame_send(&frameOut, pCtx);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_X22_ERR_SEND_COMMAND;
    }

    while (1) { // for now // TODO change this
        ypdr200_frame_t frameIn = {};
        err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_X22, pRespErrCode);
        if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) {
            if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
                return YPDR200_X22_ERR_ERROR_RESPONSE;
            }
            return YPDR200_X22_ERR_READ_RESPONSE;
        }

        if (frameIn.prolog.type != YPDR200_FRAME_TYPE_NOTIFICATION) {
            LOG_W("Unexpected frame type: expected=0x%02X, actual=0x%02X", YPDR200_FRAME_TYPE_NOTIFICATION, frameIn.prolog.type);
            ypdr200_frame_dispose(&frameIn);
            return YPDR200_X22_ERR_UNEXPECTED_FRAME_TYPE;
        }

        uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
        if (paramInLen != YPDR200_X22_NTF_PARAM_SIZE) {
            LOG_W("Unexpected param data length: expected=%d, actual=%d", YPDR200_X22_NTF_PARAM_SIZE, paramInLen);
            ypdr200_frame_dispose(&frameIn);
            return YPDR200_X22_ERR_READ_NOTIFICATION;
        }

        ypdr200_x22_ntf_param_t ntfParam;
        memcpy(ntfParam.raw, frameIn.pParamData, YPDR200_X22_NTF_PARAM_SIZE);
        if (!ypdr200_x22_ntf_param_check_crc(&ntfParam)) {
            LOG_W("ypdr200 0x22 notification CRC16 check failed for tag reply");
            ypdr200_frame_dispose(&frameIn);
            return YPDR200_X22_ERR_READ_NOTIFICATION;
        }

        if (cb != NULL) {
            cb(ntfParam, pCbUserData);
        } else {
            LOG_W("WARNING (ypdr200_x22): No callback provided for notification handling");
        }

        ypdr200_frame_dispose(&frameIn);
    }
    return YPDR200_X22_ERR_SUCCESS;
}

ypdr200_x27_req_param_t ypdr200_x27_req_param_make(uint16_t cnt) {
    return (ypdr200_x27_req_param_t) {
        .reserved = 0x22,
        .cntMsb = (uint8_t)(cnt >> 8),
        .cntLsb = (uint8_t)(cnt & 0xFF)
    };
}

int ypdr200_x27(uhfman_ctx_t* pCtx, ypdr200_x27_req_param_t param, ypdr200_resp_err_code_t* pRespErrCode, ypdr200_x22_callback cb, const void* pCbUserData) {
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_X27, YPDR200_X27_REQ_PARAM_SIZE, param.raw);
    int err = ypdr200_frame_send(&frameOut, pCtx);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_X27_ERR_SEND_COMMAND;
    }

    while (1) { // for now // TODO change this
        ypdr200_frame_t frameIn = {};
        err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_X27, pRespErrCode);
        if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) {
            if (err == YPDR200_FRAME_RECV_ERR_CKSUM_MISMATCH) {
                LOG_W("Checksum mismatch (ignoring)");
                //Flush the input buffer
                //tcflush(pCtx->fd, TCIFLUSH);
                continue;
            } else if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
                //return YPDR200_X27_ERR_ERROR_RESPONSE;
                // TODO add handler callback for error response?
                LOG_D("Received error response frame for 0x27 command, error code: 0x%02X (ignoring)", (uint8_t)(*pRespErrCode));
                continue;
            } else {
                return YPDR200_X27_ERR_READ_RESPONSE;
            }
        }

        if (frameIn.prolog.type != YPDR200_FRAME_TYPE_NOTIFICATION) {
            LOG_W("Unexpected frame type: expected=0x%02X, actual=0x%02X", YPDR200_FRAME_TYPE_NOTIFICATION, frameIn.prolog.type);
            ypdr200_frame_dispose(&frameIn);
            return YPDR200_X27_ERR_UNEXPECTED_FRAME_TYPE;
        }

        uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
        if (paramInLen != YPDR200_X22_NTF_PARAM_SIZE) {
            LOG_W("Unexpected param data length: expected=%d, actual=%d", YPDR200_X22_NTF_PARAM_SIZE, paramInLen);
            ypdr200_frame_dispose(&frameIn);
            return YPDR200_X27_ERR_READ_NOTIFICATION;
        }

        ypdr200_x22_ntf_param_t ntfParam;
        memcpy(ntfParam.raw, frameIn.pParamData, YPDR200_X22_NTF_PARAM_SIZE);
        if (!ypdr200_x22_ntf_param_check_crc(&ntfParam)) {
            LOG_W("ypdr200 0x27 notification CRC16 check failed for tag reply");
            ypdr200_frame_dispose(&frameIn);
            return YPDR200_X27_ERR_READ_NOTIFICATION;
        }

        if (cb != NULL) {
            cb(ntfParam, pCbUserData);
        } else {
            LOG_W("WARNING (ypdr200_x27): No callback provided for notification handling");
        }

        ypdr200_frame_dispose(&frameIn);
    }
    return YPDR200_X27_ERR_SUCCESS;
}
