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

#if UHFMAN_GREEDY_MODE == 1
    #define YPDR200_OUT_DELAY_MS 0
#else 
    #define YPDR200_OUT_DELAY_MS 40
#endif

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
    YPDR200_FRAME_CMD_X0C = 0x0c,
    YPDR200_FRAME_CMD_X0D = 0x0d,
    YPDR200_FRAME_CMD_X0E = 0x0e,
    YPDR200_FRAME_CMD_X11 = 0x11,
    YPDR200_FRAME_CMD_X12 = 0x12,
    YPDR200_FRAME_CMD_X22 = 0x22,
    YPDR200_FRAME_CMD_X27 = 0x27,
    YPDR200_FRAME_CMD_X28 = 0x28,
    YPDR200_FRAME_CMD_X39 = 0x39,
    YPDR200_FRAME_CMD_X49 = 0x49,
    YPDR200_FRAME_CMD_X82 = 0x82,
    YPDR200_FRAME_CMD_XAA = 0xaa,
    YPDR200_FRAME_CMD_XB6 = 0xb6,
    YPDR200_FRAME_CMD_XB7 = 0xb7,
    YPDR200_FRAME_CMD_XF1 = 0xf1,
    YPDR200_FRAME_CMD_XF2 = 0xf2,
    YPDR200_FRAME_CMD_XF3 = 0xf3,
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

static ypdr200_x49_resp_param_t __special_x49_error_response_param = {};
static ypdr200_x82_resp_param_t __special_x82_error_response_param = {};
static ypdr200_x49_resp_param_t special_x49_error_response_param_get() {
    return __special_x49_error_response_param;
}
static ypdr200_x82_resp_param_t special_x82_error_response_param_get() {
    return __special_x82_error_response_param;
}
/**
 * @param pRawParamData the copy operation performed on this buffer shall not include the `parameter` byte
 * @param parameter not really important, but we agree that it shall hold the error code in this special case
 */
static void special_x49_error_response_param_set(uint8_t* pRawParamData, uint8_t parameter) {
    memcpy(__special_x49_error_response_param.raw, pRawParamData, sizeof(__special_x49_error_response_param.raw) - 1); // - 1 because no `parameter` - this is a special case
    __special_x49_error_response_param.parameter = parameter;
}
/**
 * @param pRawParamData the copy operation performed on this buffer shall not include the `parameter` byte
 * @param parameter not really important, but we agree that it shall hold the error code in this special case
 */
static void special_x82_error_response_param_set(uint8_t* pRawParamData, uint8_t parameter) {
    memcpy(__special_x82_error_response_param.raw, pRawParamData, sizeof(__special_x82_error_response_param.raw) - 1); // - 1 because no `parameter` - this is a special case
    __special_x82_error_response_param.parameter = parameter;
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
    //actual_size_received = read(pCtx->fd, prologIn.raw, YPDR200_FRAME_PROLOG_SIZE);
    while (actual_size_received < YPDR200_FRAME_PROLOG_SIZE) {
        int poll_rv = poll(&pCtx->pollin_fd, 1, pCtx->pollin_timeout);
        if (poll_rv == -1 || poll_rv == 0) {
            LOG_W("Error polling for response prolog: poll_rv=%d", poll_rv);
#if UHFMAN_GREEDY_MODE == 0            
            //assert(0);
            LOG_E("Error polling for response prolog in greedy mode");
#endif
            return YPDR200_FRAME_RECV_ERR_READ;
        } else if (!(pCtx->pollin_fd.revents & POLLIN)) {
            LOG_W("Error polling for response prolog: pCtx->pollin_fd.revents=%d", pCtx->pollin_fd.revents);
            assert(0);
            return YPDR200_FRAME_RECV_ERR_READ;
        }
        ssize_t nread = read(pCtx->fd, prologIn.raw, YPDR200_FRAME_PROLOG_SIZE - actual_size_received);
        if (nread <= 0) {
            LOG_W("Error reading response prolog: YPD200_FRAME_PROLOG_SIZE=%d, actual_size_received=%d, nread=%d", YPDR200_FRAME_PROLOG_SIZE, actual_size_received, nread);
            assert(0);
            return YPDR200_FRAME_RECV_ERR_READ;
        }
        actual_size_received += nread;
    }
    #endif
    if (actual_size_received != YPDR200_FRAME_PROLOG_SIZE) {
        // LOG_E("Error reading response prolog: YPD200_FRAME_PROLOG_SIZE=%d, actual_size_received=%d", YPDR200_FRAME_PROLOG_SIZE, actual_size_received);
        // LOG_E_TBC("The received incomplete prolog is: ");
        // for (int i = 0; i < actual_size_received; i++) {
        //     LOG_E_CTBC("0x%02X ", prologIn.raw[i]);
        // }
        // LOG_E_CFIN("");
        // return YPDR200_FRAME_RECV_ERR_READ;
        LOG_W("Error reading response prolog: YPD200_FRAME_PROLOG_SIZE=%d, actual_size_received=%d", YPDR200_FRAME_PROLOG_SIZE, actual_size_received);
        assert(0);
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
            int poll_rv = poll(&pCtx->pollin_fd, 1, pCtx->pollin_timeout);
            if (poll_rv == -1 || poll_rv == 0) {
                LOG_W("Error polling for param data: poll_rv=%d", poll_rv);
#if UHFMAN_GREEDY_MODE == 0
                //assert(0);
                LOG_E("Error polling for param data in greedy mode");
#endif
                return YPDR200_FRAME_RECV_ERR_READ;
            } else if (!(pCtx->pollin_fd.revents & POLLIN)) {
                LOG_W("Error polling for param data: pCtx->pollin_fd.revents=%d", pCtx->pollin_fd.revents);
                assert(0);
                return YPDR200_FRAME_RECV_ERR_READ;
            }
            ssize_t nread = read(pCtx->fd, pParamIn, paramInLen - actual_size_received);
            if (nread <= 0) {
                LOG_W("Error reading param data: paramInLen=%d, actual_size_received=%d, nread=%d", paramInLen, actual_size_received, nread);
                free(pParamIn);
                assert(0);
                return YPDR200_FRAME_RECV_ERR_READ;
            }
            actual_size_received += nread;
        }
        if (actual_size_received != paramInLen) {
            LOG_W("Error reading param data: paramInLen=%d, actual_size_received=%d", paramInLen, actual_size_received);
            free(pParamIn);
            assert(0);
            return YPDR200_FRAME_RECV_ERR_READ;
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
    //actual_size_received = read(pCtx->fd, epilogIn.raw, YPDR200_FRAME_EPILOG_SIZE);
    while (actual_size_received < YPDR200_FRAME_EPILOG_SIZE) {
        int poll_rv = poll(&pCtx->pollin_fd, 1, pCtx->pollin_timeout);
        if (poll_rv == -1 || poll_rv == 0) {
            LOG_W("Error polling for response epilog: poll_rv=%d", poll_rv);
#if UHFMAN_GREEDY_MODE == 0
            assert(0);
#endif
            return YPDR200_FRAME_RECV_ERR_READ;
        } else if (!(pCtx->pollin_fd.revents & POLLIN)) {
            LOG_W("Error polling for response epilog: pCtx->pollin_fd.revents=%d", pCtx->pollin_fd.revents);
            assert(0);
            return YPDR200_FRAME_RECV_ERR_READ;
        }
        ssize_t nread = read(pCtx->fd, epilogIn.raw, YPDR200_FRAME_EPILOG_SIZE - actual_size_received);
        if (nread <= 0) {
            LOG_W("Error reading epilog: YPD200_FRAME_EPILOG_SIZE=%d, actual_size_received=%d, nread=%d", YPDR200_FRAME_EPILOG_SIZE, actual_size_received, nread);
            if (pParamIn != (uint8_t*)0) {
                free(pParamIn);
            }
            assert(0);
            return YPDR200_FRAME_RECV_ERR_READ;
        }
        actual_size_received += nread;
    }
    #endif
    if (actual_size_received != YPDR200_FRAME_EPILOG_SIZE) {
        LOG_E("Error reading epilog: YPD200_FRAME_EPILOG_SIZE=%d, actual_size_received=%d", YPDR200_FRAME_EPILOG_SIZE, actual_size_received);
        if (pParamIn != (uint8_t*)0) {
            free(pParamIn);
        }
        assert(0);
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
        LOG_D("Checksum mismatch: expected=0x%02X, actual=0x%02X", checksum, epilogIn.checksum); // TODO don't ignore
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

    if (! (expectedCmd == YPDR200_FRAME_CMD_X27 && frameIn.prolog.cmd == YPDR200_FRAME_CMD_X22)) { // We receive x22 notifications in response to x27 command as well, that's where this condition comes from (see YPD-R200 user protocol reference)
        if (frameIn.prolog.cmd != (uint8_t)expectedCmd) {
            if (frameIn.prolog.cmd == (uint8_t)YPDR200_FRAME_CMD_XFF) {
                uint16_t respParamLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
                if (pParamIn == NULL) {
                    LOG_W("Received error response frame, but resp param is NULL (unlikely)\n");
                    return YPDR200_FRAME_RECV_ERR_READ;
                }
                if (respParamLen != 1) {
                    if ((expectedCmd != YPDR200_FRAME_CMD_X49) && (expectedCmd != YPDR200_FRAME_CMD_X82)) { // normally we expect respParamLen to strictly always be 1 for error response frames...
                        if (pParamIn != (uint8_t*)0) {
                            free(pParamIn);
                        }
                        LOG_W("Received error response frame, but resp param had unexpected length %u\n", respParamLen);
                        return YPDR200_FRAME_RECV_ERR_READ;
                    } else if (expectedCmd == YPDR200_FRAME_CMD_X49) { // ...but in case of x49, it is possible that we receive a special error response (see YPD-R200 user protocol reference) and need to set a file-scope variable for that
                        if (respParamLen != YPDR200_X49_RESP_PARAM_SIZE) { // `parameter` is excluded, but the dat is prepended with the error code byte instead, so we still get an expected resp param length of YPDR200_X49_RESP_PARAM_SIZE bytes
                            if (pParamIn != (uint8_t*)0) {
                                free(pParamIn);
                            }
                            LOG_W("Received special x49 error response frame, but resp param had unexpected length %u\n", respParamLen);
                            return YPDR200_FRAME_RECV_ERR_READ;
                        }
                        // pParamIn[0] is the error code and is already handled few lines later, however we will still save it as parameter - this field would be otherwise unused, but it makes to put the error code in there
                        special_x49_error_response_param_set(pParamIn + 1, pParamIn[0]);
                    } else if (expectedCmd == YPDR200_FRAME_CMD_X82) { // ...same for x82
                        if (respParamLen != YPDR200_X82_RESP_PARAM_SIZE) { // `parameter` is excluded, but the dat is prepended with the error code byte instead, so we still get an expected resp param length of YPDR200_X82_RESP_PARAM_SIZE bytes
                            if (pParamIn != (uint8_t*)0) {
                                free(pParamIn);
                            }
                            LOG_W("Received special x82 error response frame, but resp param had unexpected length %u\n", respParamLen);
                            return YPDR200_FRAME_RECV_ERR_READ;
                        }
                        // pParamIn[0] is the error code and is already handled few lines later, however we will still save it as parameter - this field would be otherwise unused, but it makes to put the error code in there
                        special_x82_error_response_param_set(pParamIn + 1, pParamIn[0]);
                    }
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

    // Timeout to prevent overwhelming the device
    msleep(YPDR200_OUT_DELAY_MS);

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
    //actual_size_transmitted = write(pCtx->fd, pDataOut, dataOutLen);
    while (actual_size_transmitted < dataOutLen) {
        //msleep(YPDR200_OUT_DELAY_MS);
        int poll_rv = poll(&pCtx->pollout_fd, 1, pCtx->pollout_timeout);
        if (poll_rv == -1 || poll_rv == 0) {
            LOG_W("Error polling for frame write: pollout_rv=%d", poll_rv);
            assert(0);
            return YPDR200_FRAME_SEND_ERR_SEND;
        } else if (!(pCtx->pollout_fd.revents & POLLOUT)) {
            LOG_W("Error polling for frame write: pCtx->pollout_fd.revents=%d", pCtx->pollout_fd.revents);
            assert(0);
            return YPDR200_FRAME_SEND_ERR_SEND;
        }
        ssize_t nwritten = write(pCtx->fd, pDataOut, dataOutLen - actual_size_transmitted);
        if (nwritten <= 0) {
            LOG_W("Error sending cmd 0x%02X: dataOutLen=%d, actual_size_transmitted=%d, nwritten=%d", pFrameSnd->prolog.cmd, dataOutLen, actual_size_transmitted, nwritten);
            free(pDataOut);
            assert(0);
            return YPDR200_FRAME_SEND_ERR_SEND;
        }
        actual_size_transmitted += nwritten;
    }
    if (actual_size_transmitted != dataOutLen) {
        LOG_W("Error sending cmd 0x%02X: dataOutLen=%d, actual_size_transmitted=%d", pFrameSnd->prolog.cmd, dataOutLen, actual_size_transmitted);
        free(pDataOut);
        assert(0);
        return YPDR200_FRAME_SEND_ERR_SEND;
    }
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
        //ypdr200_x0b_resp_param_t respParam = {};
        memcpy(pRespParam->hdr.raw, pRawParamData, YPDR200_X0B_RESP_PARAM_HDR_SIZE);
        //*pRespParam = respParam;
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
    LOG_V("Copying mask data of length %d bytes", pRespParam->hdr.maskLen >> 3);
    memcpy(pRespParam->pMask, pMask, pRespParam->hdr.maskLen >> 3);
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
    uint8_t mask_nbytes = respParam.hdr.maskLen >> 3;
    if (respParam.hdr.maskLen & 0x07) {
        mask_nbytes++;
        LOG_W("Received mask length is not a multiple of 8: maskLen=%d. Rounding mask_nbytes up to %d", respParam.hdr.maskLen, mask_nbytes);
    }
    if (mask_nbytes + YPDR200_X0B_RESP_PARAM_HDR_SIZE != paramInLen) {
        LOG_W("Unexpected mask length: expected mask_nbytes=%d (respParam.hdr.maskLen=%d), actual=%d", mask_nbytes, respParam.hdr.maskLen, paramInLen - YPDR200_X0B_RESP_PARAM_HDR_SIZE);
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

int ypdr200_xb6(uhfman_ctx_t* pCtx, uint16_t txPower, ypdr200_resp_err_code_t* pRespErrCode) {
    uint8_t txPowerMsb = (uint8_t)(txPower >> 8);
    uint8_t txPowerLsb = (uint8_t)(txPower & 0xFF);
    uint8_t paramData[2] = {txPowerMsb, txPowerLsb};
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_XB6, 2U, paramData);

    int err = ypdr200_frame_send(&frameOut, pCtx);
    // No need to call ypdr200_frame_dispose, as paramData is allocated statically

    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_XB6_ERR_SEND_COMMAND;
    }

    ypdr200_frame_t frameIn = {};
    err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_XB6, pRespErrCode);
    if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) {
        if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
            return YPDR200_XB6_ERR_ERROR_RESPONSE;
        }
        return YPDR200_XB6_ERR_READ_RESPONSE;
    }

    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
    if (paramInLen != 1) {
        LOG_W("Unexpected param data length: expected=1, actual=%d", paramInLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_XB6_ERR_READ_RESPONSE;
    }

    if (frameIn.pParamData[0] != 0U) {
        LOG_W("Unexpected 0xb6 command response data: %02x. We were expecting 0x00 for success.", frameIn.pParamData[0]);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_XB6_ERR_READ_RESPONSE;
    }

    ypdr200_frame_dispose(&frameIn);
    return YPDR200_XB6_ERR_SUCCESS;
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

int ypdr200_x22(uhfman_ctx_t* pCtx, ypdr200_resp_err_code_t* pRespErrCode, ypdr200_x22_callback cb, void* pCbUserData) {
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_X22, 0U, NULL);
    int err = ypdr200_frame_send(&frameOut, pCtx);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_X22_ERR_SEND_COMMAND;
    }

    //while (1) { // for now // TODO change this
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

    ///<debug>
    if (ntfParam.epc[0] == 0xAA) {
        LOG_W("Received notification with EPC starting with AA");
    } else {
        LOG_D("Received notification with EPC starting with %02X", ntfParam.epc[0]);
    }
    ///</debug>

    if (cb != NULL) {
        cb(ntfParam, pCbUserData);
    } else {
        LOG_W("WARNING (ypdr200_x22): No callback provided for notification handling");
    }

    ypdr200_frame_dispose(&frameIn);
    //}
    return YPDR200_X22_ERR_SUCCESS;
}

ypdr200_x27_req_param_t ypdr200_x27_req_param_make(uint16_t cnt) {
    return (ypdr200_x27_req_param_t) {
        .reserved = 0x22,
        .cntMsb = (uint8_t)(cnt >> 8),
        .cntLsb = (uint8_t)(cnt & 0xFF)
    };
}

int ypdr200_x27(uhfman_ctx_t* pCtx, ypdr200_x27_req_param_t param, ypdr200_resp_err_code_t* pRespErrCode, ypdr200_x22_callback cb, void* pCbUserData, size_t timeout_us) {
    // <<< implement timeout
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_X27, YPDR200_X27_REQ_PARAM_SIZE, param.raw);
    int err = ypdr200_frame_send(&frameOut, pCtx);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_X27_ERR_SEND_COMMAND;
    }

    struct timespec ts;
    assert(0 == clock_gettime(CLOCK_MONOTONIC, &ts));
    uint64_t start_us = (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;

    while (1) { // until timeout
        LOG_D("Waiting for 0x27 notification frame...");
        ypdr200_frame_t frameIn = {};
        err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_X27, pRespErrCode);
        if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) {
            if (err == YPDR200_FRAME_RECV_ERR_CKSUM_MISMATCH) {
                LOG_D("Checksum mismatch (ignoring)"); // TODO Can we fix those??
                //Flush the input buffer
                //tcflush(pCtx->fd, TCIFLUSH);

                // Check timeout
                assert(0 == clock_gettime(CLOCK_MONOTONIC, &ts));
                uint64_t now_us = (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
                uint64_t timeout_elapsed = now_us - start_us;
                LOG_D("Timeout elapsed: %lu us", timeout_elapsed);
                if (timeout_elapsed >= timeout_us) {
                    break;
                }
                continue;
            } else if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
                //return YPDR200_X27_ERR_ERROR_RESPONSE;
                // TODO add handler callback for error response?
                LOG_D("Received error response frame for 0x27 command, error code: 0x%02X (ignoring)", (uint8_t)(*pRespErrCode));
                
                // Check timeout
                assert(0 == clock_gettime(CLOCK_MONOTONIC, &ts));
                uint64_t now_us = (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
                uint64_t timeout_elapsed = now_us - start_us;
                LOG_D("Timeout elapsed: %lu us", timeout_elapsed);
                if (timeout_elapsed >= timeout_us) {
                    break;
                }
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

        // Check timeout
        assert(0 == clock_gettime(CLOCK_MONOTONIC, &ts));
        uint64_t now_us = (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
        uint64_t timeout_elapsed = now_us - start_us;
        LOG_D("Timeout elapsed: %lu us", timeout_elapsed);
        if (timeout_elapsed >= timeout_us) {
            break;
        }
    }
    return YPDR200_X27_ERR_SUCCESS;
}

int ypdr200_x28(uhfman_ctx_t* pCtx, ypdr200_resp_err_code_t* pRespErrCode) {
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_X28, 0U, NULL);
    int err = ypdr200_frame_send(&frameOut, pCtx);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_X28_ERR_SEND_COMMAND;
    }

    ypdr200_frame_t frameIn = {};
    err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_X28, pRespErrCode);
    if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) {
        if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
            return YPDR200_X28_ERR_ERROR_RESPONSE;
        }
        return YPDR200_X28_ERR_READ_RESPONSE;
    }

    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
    if (paramInLen != 1) {
        LOG_W("Unexpected param data length: expected=1, actual=%d", paramInLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X28_ERR_READ_RESPONSE;
    }

    if (frameIn.pParamData[0] != 0U) {
        LOG_W("Unexpected 0x28 command response data: %02x. We were expecting 0x00 for success.", frameIn.pParamData[0]);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X28_ERR_READ_RESPONSE;
    }

    ypdr200_frame_dispose(&frameIn);
    return YPDR200_X28_ERR_SUCCESS;
}

int ypdr200_x12(uhfman_ctx_t* pCtx, uint8_t mode, ypdr200_resp_err_code_t* pRespErrCode) {
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_X12, (uint16_t)1U, &mode);
    int err = ypdr200_frame_send(&frameOut, pCtx);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_X12_ERR_SEND_COMMAND;
    }

    ypdr200_frame_t frameIn = {};
    err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_X12, pRespErrCode);
    if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) {
        if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
            return YPDR200_X12_ERR_ERROR_RESPONSE;
        }
        return YPDR200_X12_ERR_READ_RESPONSE;
    }

    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
    if (paramInLen != 1) {
        LOG_W("Unexpected param data length: expected=1, actual=%d", paramInLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X12_ERR_READ_RESPONSE;
    }

    if (frameIn.pParamData[0] != 0U) {
        LOG_W("Unexpected 0x12 command response data: %02x. We were expecting 0x00 for success.", frameIn.pParamData[0]);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X12_ERR_READ_RESPONSE;
    }

    ypdr200_frame_dispose(&frameIn);
    return YPDR200_X12_ERR_SUCCESS;
}

ypdr200_x0c_req_param_t ypdr200_x0c_req_param_make(ypdr200_x0c_req_param_hdr_t hdr, uint8_t* pMask) {
    return (ypdr200_x0c_req_param_t) {
        .hdr = hdr,
        .pMask = pMask
    };
}

void ypdr200_x0c_req_param_dispose(ypdr200_x0c_req_param_t* pReqParam) {
    if (pReqParam->pMask != (uint8_t*)0) {
        free(pReqParam->pMask);
    }
}

/**
 * @param ppRaw_out Address of the allocated memory block will be stored here
 * @returns 0 on success, -1 on error (check errno)
 * @note The caller is responsible for freeing the allocated memory block at *ppRaw_out
 */
static int ypdr200_x0c_req_param_raw(ypdr200_x0c_req_param_t* pReqParam, uint8_t** ppRaw_out) {
    uint8_t mask_nbytes = pReqParam->hdr.maskLen >> 3;
    if (pReqParam->hdr.maskLen & 0x07) {
        mask_nbytes++;
    }
    *ppRaw_out = (uint8_t*) malloc(mask_nbytes + YPDR200_X0C_REQ_PARAM_HDR_SIZE);
    if (*ppRaw_out == (uint8_t*)0) {
        errno = ENOMEM;
        return -1;
    }
    memcpy(*ppRaw_out, pReqParam->hdr.raw, YPDR200_X0C_REQ_PARAM_HDR_SIZE);
    memcpy((*ppRaw_out) + YPDR200_X0C_REQ_PARAM_HDR_SIZE, pReqParam->pMask, mask_nbytes);
    return 0;
}

int ypdr200_x0c(uhfman_ctx_t* pCtx, ypdr200_x0c_req_param_t* pReqParam, ypdr200_resp_err_code_t* pRespErrCode) {
    uint8_t *pRawFrameOut = (uint8_t*)0; //TODO: Avoid copying header data?
    if (-1 == ypdr200_x0c_req_param_raw(pReqParam, &pRawFrameOut)) {
        if (errno == ENOMEM) {
            LOG_W("Ran out of memory while preparing 0x0C command frame!");
        }
        return YPDR200_X0C_ERR_SEND_COMMAND;
    }
    uint8_t mask_nbytes = pReqParam->hdr.maskLen >> 3;
    if (pReqParam->hdr.maskLen & 0x07) {
        mask_nbytes++;
        LOG_W("Mask length is not a multiple of 8, rounding up mask_nbytes to %d", mask_nbytes);
    }
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_X0C, mask_nbytes + YPDR200_X0C_REQ_PARAM_HDR_SIZE, pRawFrameOut);
    
    int err = ypdr200_frame_send(&frameOut, pCtx);
    free(pRawFrameOut);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_X0C_ERR_SEND_COMMAND;
    }

    ypdr200_frame_t frameIn = {};
    err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_X0C, pRespErrCode);
    if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) {
        if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
            return YPDR200_X0C_ERR_ERROR_RESPONSE;
        }
        return YPDR200_X0C_ERR_READ_RESPONSE;
    }

    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
    if (paramInLen != 1) {
        LOG_W("Unexpected param data length: expected=1, actual=%d", paramInLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X0C_ERR_READ_RESPONSE;
    }

    if (frameIn.pParamData[0] != 0U) {
        LOG_W("Unexpected 0x0C command response data: %02x. We were expecting 0x00 for success.");
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X0C_ERR_READ_RESPONSE;
    }

    ypdr200_frame_dispose(&frameIn);
    return YPDR200_X0C_ERR_SUCCESS;
}

int ypdr200_x0e(uhfman_ctx_t* pCtx, ypdr200_x0e_req_param_t param, ypdr200_resp_err_code_t* pRespErrCode) {
    // param.raw is uint16_t so we need to handle that appropriately
    uint8_t paramData[2] = {param.raw >> 8, param.raw & 0xFF};
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_X0E, 2, paramData);
    int err = ypdr200_frame_send(&frameOut, pCtx);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_X0E_ERR_SEND_COMMAND;
    }

    ypdr200_frame_t frameIn = {};
    err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_X0E, pRespErrCode);
    if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) {
        if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
            return YPDR200_X0E_ERR_ERROR_RESPONSE;
        }
        return YPDR200_X0E_ERR_READ_RESPONSE;
    }

    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
    if (paramInLen != 1) {
        LOG_W("Unexpected param data length: expected=1, actual=%d", paramInLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X0E_ERR_READ_RESPONSE;
    }

    if (frameIn.pParamData[0] != 0U) {
        LOG_W("Unexpected 0x0E command response data: %02x. We were expecting 0x00 for success.");
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X0E_ERR_READ_RESPONSE;
    }

    ypdr200_frame_dispose(&frameIn);
    return YPDR200_X0E_ERR_SUCCESS;
}

void ypdr200_jmdr_dispose(ypdr200_xf3_resp_param_t* pRespParam) {
    if (pRespParam->pJmr != (uint8_t*)0) {
        free(pRespParam->pJmr);
    }
}

uint16_t ypdr200_jmdr_get_jmr_len(ypdr200_xf3_resp_param_t* pRespParam) {
    if (pRespParam->pJmr == (uint8_t*)0) {
        return 0;
    } else if (pRespParam->hdr.ch_l > pRespParam->hdr.ch_h) {
        return 0;
    }
    return pRespParam->hdr.ch_h - pRespParam->hdr.ch_l + 1;
}

/**
 * @warning Need to check outside of this function if the pointer to be used is valid
 */
static void ypdr200_jmdr_set_hdr_from_raw(uint8_t* pRawParamData, ypdr200_xf3_resp_param_t* pRespParam) {
    if (pRespParam != NULL) {
        pRespParam->hdr.raw = (((uint16_t)pRawParamData[0]) << 8) | (uint16_t)pRawParamData[1];
    }
}

/**
 * @warning Need to check outside of this function if the pointer to be used is valid
 * @returns 0 on success, -1 on error (check errno)
 */
static int ypdr200_jmdr_set_jmr(uint8_t* pJmr, ypdr200_xf3_resp_param_t* pRespParam) {
    uint16_t jmrLen = ypdr200_jmdr_get_jmr_len(pRespParam);
    pRespParam->pJmr = malloc((size_t)jmrLen);
    if (pRespParam->pJmr == (uint8_t*)0) {
        errno = ENOMEM;
        return -1;
    }
    memcpy(pRespParam->pJmr, pJmr, jmrLen);
    return 0;
}

int ypdr200_xf3(uhfman_ctx_t* pCtx, ypdr200_xf3_resp_param_t* pRespParam_out, ypdr200_resp_err_code_t* pRespErrCode) {
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_XF3, 0U, NULL);
    int err = ypdr200_frame_send(&frameOut, pCtx);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_XF3_ERR_SEND_COMMAND;
    }

    ypdr200_frame_t frameIn = {};
    err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_XF3, pRespErrCode);
    if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) {
        if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
            return YPDR200_XF3_ERR_ERROR_RESPONSE;
        }
        return YPDR200_XF3_ERR_READ_RESPONSE;
    }

    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
    if (paramInLen < 1) {
        LOG_W("Unexpected param data length: expected>0, actual=%u", paramInLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_XF3_ERR_READ_RESPONSE;
    }

    ypdr200_xf3_resp_param_t respParam = {};
    ypdr200_jmdr_set_hdr_from_raw(frameIn.pParamData, &respParam);
    uint16_t expectedParamInLen = sizeof(respParam.hdr.raw) + ypdr200_jmdr_get_jmr_len(&respParam);
    if (paramInLen != expectedParamInLen) {
        LOG_W("Unexpected JMR length: expected=%u, actual=%u", expectedParamInLen, paramInLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_XF3_ERR_READ_RESPONSE;
    }

    if (-1 == ypdr200_jmdr_set_jmr(frameIn.pParamData + sizeof(respParam.hdr.raw), &respParam)) {
        uhfman_debug_errno(); // print errno
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_XF3_ERR_READ_RESPONSE;
    }

    ypdr200_frame_dispose(&frameIn);

    *pRespParam_out = respParam;
    return YPDR200_XF3_ERR_SUCCESS;
}

// TODO DRY (compare with ypdr200_xf3)
int ypdr200_xf2(uhfman_ctx_t* pCtx, ypdr200_xf2_resp_param_t* pRespParam_out, ypdr200_resp_err_code_t* pRespErrCode) {
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_XF2, 0U, NULL);
    int err = ypdr200_frame_send(&frameOut, pCtx);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_XF2_ERR_SEND_COMMAND;
    }

    ypdr200_frame_t frameIn = {};
    err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_XF2, pRespErrCode);
    if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) {
        if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
            return YPDR200_XF2_ERR_ERROR_RESPONSE;
        }
        return YPDR200_XF2_ERR_READ_RESPONSE;
    }

    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
    if (paramInLen < 1) {
        LOG_W("Unexpected param data length: expected>0, actual=%u", paramInLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_XF2_ERR_READ_RESPONSE;
    }

    ypdr200_xf2_resp_param_t respParam = {};
    ypdr200_jmdr_set_hdr_from_raw(frameIn.pParamData, &respParam);
    uint16_t expectedParamInLen = sizeof(respParam.hdr.raw) + ypdr200_jmdr_get_jmr_len(&respParam);
    if (paramInLen != expectedParamInLen) {
        LOG_W("Unexpected JMR length: expected=%u, actual=%u", expectedParamInLen, paramInLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_XF2_ERR_READ_RESPONSE;
    }

    if (-1 == ypdr200_jmdr_set_jmr(frameIn.pParamData + sizeof(respParam.hdr.raw), &respParam)) {
        uhfman_debug_errno(); // print errno
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_XF2_ERR_READ_RESPONSE;
    }

    ypdr200_frame_dispose(&frameIn);

    *pRespParam_out = respParam;
    return YPDR200_XF2_ERR_SUCCESS;
}

void ypdr200_x39_resp_param_dispose(ypdr200_x39_resp_param_t* pRespParam) {
    if (pRespParam->body.pData != (uint8_t*)0) {
        free(pRespParam->body.pData);
    }
}

static void ypdr200_x39_resp_param_set_hdr_from_raw(uint8_t* pRawParamData, ypdr200_x39_resp_param_t* pRespParam) {
    if (pRespParam != NULL) {
        memcpy(pRespParam->hdr.raw, pRawParamData, YPDR200_X39_RESP_PARAM_HDR_SIZE);
    }
}

static int ypdr200_x39_resp_param_set_data(uint8_t* pData, uint16_t dataLen, ypdr200_x39_resp_param_t* pRespParam) {
    pRespParam->body.dataLen = dataLen;
    pRespParam->body.pData = malloc((size_t)dataLen);
    if (pRespParam->body.pData == (uint8_t*)0) {
        errno = ENOMEM;
        return -1;
    }
    memcpy(pRespParam->body.pData, pData, dataLen);
    return 0;
}

int ypdr200_x39(uhfman_ctx_t* pCtx, ypdr200_x39_req_param_t param, ypdr200_x39_resp_param_t* pRespParam_out, ypdr200_resp_err_code_t* pRespErrCode) {
    uint16_t requestedDataReadLength = (((uint16_t)param.dl[0]) << 8) | (uint16_t)param.dl[1];
    
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_X39, YPDR200_X39_REQ_PARAM_SIZE, param.raw);
    int err = ypdr200_frame_send(&frameOut, pCtx);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_X39_ERR_SEND_COMMAND;
    }

    ypdr200_frame_t frameIn = {};
    err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_X39, pRespErrCode);
    if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) {
        if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
            return YPDR200_X39_ERR_ERROR_RESPONSE;
        }
        return YPDR200_X39_ERR_READ_RESPONSE;
    }

    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
    if (paramInLen < YPDR200_X39_RESP_PARAM_HDR_SIZE) {
        LOG_W("Unexpected param data length: expected>=%d,  actual=%u", YPDR200_X39_RESP_PARAM_HDR_SIZE, paramInLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X39_ERR_READ_RESPONSE;
    }

    ypdr200_x39_resp_param_t respParam = {};
    ypdr200_x39_resp_param_set_hdr_from_raw(frameIn.pParamData, &respParam);
    
    if (respParam.hdr.ul + 1 != YPDR200_X39_RESP_PARAM_HDR_SIZE) {// TODO check if can be made flexible depending on ul
        LOG_W("Unexpected ul: expected=%d, actual=%d", YPDR200_X39_RESP_PARAM_HDR_SIZE, respParam.hdr.ul);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X39_ERR_READ_RESPONSE;
    }

    uint16_t dataLen = paramInLen - YPDR200_X39_RESP_PARAM_HDR_SIZE;

    if (requestedDataReadLength != dataLen) {
        LOG_W("Unexpected data length: expected=%d, actual=%d", requestedDataReadLength, dataLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X39_ERR_READ_RESPONSE;
    }

    if (-1 == ypdr200_x39_resp_param_set_data(frameIn.pParamData + respParam.hdr.ul, dataLen, &respParam)) {
        uhfman_debug_errno(); // print errno
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X39_ERR_READ_RESPONSE;
    }

    ypdr200_frame_dispose(&frameIn);

    *pRespParam_out = respParam;
    return YPDR200_X39_ERR_SUCCESS;
}

ypdr200_x49_req_param_hdr_t ypdr200_x49_req_param_hdr_make(uint16_t sa, uint16_t dl, uint8_t memBank, const uint8_t ap[4]) {
    return (ypdr200_x49_req_param_hdr_t) {
        .sa = {sa >> 8, sa & 0xFF},
        .dl = {dl >> 8, dl & 0xFF},
        .memBank = memBank,
        .ap = {ap[0], ap[1], ap[2], ap[3]} // @note we are assuming ap is MSB-first as noted in the header file
    };
}

ypdr200_x49_req_param_t ypdr200_x49_req_param_make(ypdr200_x49_req_param_hdr_t hdr, uint8_t* pDt) {
    return (ypdr200_x49_req_param_t) {
        .hdr = hdr,
        .pDt = pDt
    };
}

void ypdr200_x49_req_param_dispose(ypdr200_x49_req_param_t* pReqParam) {
    if (pReqParam->pDt != (uint8_t*)0) {
        free(pReqParam->pDt);
    }
}

static uint16_t ypdr200_x49_req_param_get_dl(ypdr200_x49_req_param_t* pReqParam) {
    return (((uint16_t)pReqParam->hdr.dl[0]) << 8) | (uint16_t)pReqParam->hdr.dl[1];
}

/**
 * @param ppRaw_out Address of the allocated memory block will be stored here
 * @returns 0 on success, -1 on error (check errno)
 * @note The caller is responsible for freeing the allocated memory block at *ppRaw_out
 */
static int ypdr200_x49_req_param_raw(ypdr200_x49_req_param_t* pReqParam, uint8_t** ppRaw_out) {
    uint16_t dl = ypdr200_x49_req_param_get_dl(pReqParam);
    size_t data_nbytes = dl << 1;
    *ppRaw_out = (uint8_t*) malloc(data_nbytes + YPDR200_X49_REQ_PARAM_HDR_SIZE);
    if (*ppRaw_out == (uint8_t*)0) {
        errno = ENOMEM;
        return -1;
    }
    memcpy(*ppRaw_out, pReqParam->hdr.raw, YPDR200_X49_REQ_PARAM_HDR_SIZE);
    memcpy((*ppRaw_out) + YPDR200_X49_REQ_PARAM_HDR_SIZE, pReqParam->pDt, data_nbytes);
    return 0;
}

int ypdr200_x49(uhfman_ctx_t* pCtx, ypdr200_x49_req_param_t param, ypdr200_x49_resp_param_t* pRespParam_out, ypdr200_resp_err_code_t* pRespErrCode) {
    // TODO generalize the procedure so that variable-length params can be used more easily and without repeating code
    uint8_t* pRawFrameOut = (uint8_t*)0; //TODO: Avoid copying header data?
    if (-1 == ypdr200_x49_req_param_raw(&param, &pRawFrameOut)) {
        if (errno == ENOMEM) {
            LOG_W("Ran out of memory while preparing 0x49 command frame!");
        }
        return YPDR200_X49_ERR_SEND_COMMAND;
    }
    uint16_t reqDl = ypdr200_x49_req_param_get_dl(&param);
    size_t req_data_nbytes = reqDl << 1;
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_X49, YPDR200_X49_REQ_PARAM_HDR_SIZE + req_data_nbytes, pRawFrameOut);
    int err = ypdr200_frame_send(&frameOut, pCtx);
    free(pRawFrameOut);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_X49_ERR_SEND_COMMAND;
    }

    ypdr200_frame_t frameIn = {};
    err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_X49, pRespErrCode);
    if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) { // TODO try not to repeat these lines of code
        if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
            if (*pRespErrCode == YPDR200_RESP_ERR_CODE_ACCESS_FAIL) {
                // Handle the special x49 error response
                *pRespParam_out = special_x49_error_response_param_get();
                if (pRespParam_out->ul + 2 != YPDR200_X49_RESP_PARAM_SIZE) { // +2 because it is the special case - error code is before the ul byte and replaces the parameter byte
                    LOG_W("Unexpected ul: expected=%d, actual=%d", YPDR200_X49_RESP_PARAM_SIZE - 2, pRespParam_out->ul);
                    ypdr200_frame_dispose(&frameIn);
                    return YPDR200_X49_ERR_READ_RESPONSE;
                } 
            }
            return YPDR200_X49_ERR_ERROR_RESPONSE;
        }
        return YPDR200_X49_ERR_READ_RESPONSE;
    }

    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
    if (paramInLen != YPDR200_X49_RESP_PARAM_SIZE) {
        LOG_W("Unexpected param data length: expected=%d, actual=%d", YPDR200_X49_RESP_PARAM_SIZE, paramInLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X49_ERR_READ_RESPONSE;
    }

    ypdr200_x49_resp_param_t respParam = {};
    memcpy(respParam.raw, frameIn.pParamData, YPDR200_X49_RESP_PARAM_SIZE);

    if (respParam.parameter != 0U) { // TODO check for consistency with other functions using the same pattern for success check?
        LOG_W("Unexpected 0x49 command response data: %02x. We were expecting 0x00 for success.", respParam.parameter);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X49_ERR_READ_RESPONSE;
    }

    ypdr200_frame_dispose(&frameIn);

    *pRespParam_out = respParam;
    return YPDR200_X49_ERR_SUCCESS;
}

int ypdr200_x82(uhfman_ctx_t* pCtx, ypdr200_x82_req_param_t param, ypdr200_x82_resp_param_t* pRespParam_out, ypdr200_resp_err_code_t* pRespErrCode) {
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_X82, YPDR200_X82_REQ_PARAM_SIZE, param.raw);
    int err = ypdr200_frame_send(&frameOut, pCtx);
    if (err != YPDR200_FRAME_SEND_ERR_SUCCESS) {
        return YPDR200_X82_ERR_SEND_COMMAND;
    }

    ypdr200_frame_t frameIn = {};
    err = ypdr200_frame_recv(&frameIn, pCtx, YPDR200_FRAME_CMD_X82, pRespErrCode);
    if (err != YPDR200_FRAME_RECV_ERR_SUCCESS) { // TODO try not to repeat these lines of code
        if (err == YPDR200_FRAME_RECV_ERR_GOT_ERR_RESPONSE_FRAME) {
            if (*pRespErrCode == YPDR200_RESP_ERR_CODE_ACCESS_FAIL) {
                // Handle the special x49 error response
                *pRespParam_out = special_x82_error_response_param_get();
                if (pRespParam_out->ul + 2 != YPDR200_X82_RESP_PARAM_SIZE) { // +2 because it is the special case - error code is before the ul byte and replaces the parameter byte
                    LOG_W("Unexpected ul: expected=%d, actual=%d", YPDR200_X82_RESP_PARAM_SIZE - 2, pRespParam_out->ul);
                    ypdr200_frame_dispose(&frameIn);
                    return YPDR200_X82_ERR_READ_RESPONSE;
                } 
            }
            return YPDR200_X82_ERR_ERROR_RESPONSE;
        }
        return YPDR200_X82_ERR_READ_RESPONSE;
    }

    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&frameIn.prolog);
    if (paramInLen != YPDR200_X82_RESP_PARAM_SIZE) {
        LOG_W("Unexpected param data length: expected=%d, actual=%d", YPDR200_X82_RESP_PARAM_SIZE, paramInLen);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X82_ERR_READ_RESPONSE;
    }

    ypdr200_x82_resp_param_t respParam = {};
    memcpy(respParam.raw, frameIn.pParamData, YPDR200_X82_RESP_PARAM_SIZE);

    if (respParam.parameter != 0U) { // TODO check for consistency with other functions using the same pattern for success check?
        LOG_W("Unexpected 0x82 command response data: %02x. We were expecting 0x00 for success.", respParam.parameter);
        ypdr200_frame_dispose(&frameIn);
        return YPDR200_X82_ERR_READ_RESPONSE;
    }

    ypdr200_frame_dispose(&frameIn);

    *pRespParam_out = respParam;
    return YPDR200_X82_ERR_SUCCESS;
}