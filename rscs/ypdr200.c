#include "ypdr200.h"
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
//#include <stdio.h>
//#include <fcntl.h>
#include <unistd.h>

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
} ypdr200_frame_prolog_t;

#define YPDR200_FRAME_EPILOG_SIZE 2
typedef union _ypdr200_frame_epilog {
    struct {
        uint8_t checksum;
        uint8_t end;
    };
    uint8_t raw[YPDR200_FRAME_EPILOG_SIZE];
} ypdr200_frame_epilog_t;

typedef struct _ypdr200_frame {
    ypdr200_frame_prolog_t prolog;
    ypdr200_frame_epilog_t epilog;
    uint8_t* pParamData;
} ypdr200_frame_t;

static uint16_t ypdr200_frame_prolog_get_param_length(const ypdr200_frame_prolog_t* pProlog) {
    uint16_t paramLen = (uint16_t)(pProlog->paramLengthMsb);
    paramLen = (paramLen << 8) || (uint16_t)(pProlog->paramLengthLsb);
    return paramLen;
}

static uint8_t ypdr200_frame_get_checksum(const ypdr200_frame_t* pFrame) {
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
    YPDR200_FRAME_CMD_X03 = 0x03
} ypdr200_frame_cmd_t;

/**
 * @note The buffer pointed by pParamData will be directly referenced by the frame structure
 */
static void ypdr200_frame_set_param_data(ypdr200_frame_t* pFrame, uint8_t* pParamData) {
    pFrame->pParamData = pParamData;
}

static void ypdr200_frame_set_checksum(ypdr200_frame_t* pFrame) {
    pFrame->epilog.checksum = ypdr200_frame_get_checksum(pFrame);
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

// TODO: Separate response handling to a common static function
int ypdr200_x03(uhfman_ctx_t* pCtx, ypdr200_x03_param_t infoType, char** ppcInfo_out) {
    uint8_t param = (uint8_t)infoType;
    ypdr200_frame_t frameOut = ypdr200_frame_construct(YPDR200_FRAME_TYPE_COMMAND, YPDR200_FRAME_CMD_X03, 1U, &param);
    uint8_t* pDataOut = ypdr200_frame_raw(&frameOut);
    uint32_t dataOutLen = ypdr200_frame_length(&frameOut);
    assert(dataOutLen == 8U);
    assert(pDataOut[0] == 0xAA);
    assert(pDataOut[1] == 0x00);
    assert(pDataOut[2] == 0x03);
    assert(pDataOut[3] == 0x00);
    assert(pDataOut[4] == 0x01);
    assert(pDataOut[5] == (uint8_t)infoType);
    assert(pDataOut[6] == 0x04 + (uint8_t)infoType);
    assert(pDataOut[7] == 0xDD);

    int actual_size_transmitted = 0;
#if YPDR200_INTERFACE_TYPE == YPDR200_INTERFACE_TYPE_LIBUSB
    int rv = libusb_bulk_transfer(pCtx->handle, YPDR200_BULK_ENDPOINT_ADDR_OUT, pDataOut, dataOutLen, &actual_size_transmitted, YPDR200_BULK_TRANSFER_TIMEOUT_MS);
    if (rv != 0) {
        fprintf(stderr, "Error sending cmd 0x03: %s (%d)\n", libusb_error_name(rv), rv);
        return YPDR200_X03_ERR_SEND_COMMAND;
    }
#elif YPDR200_INTERFACE_TYPE == YPDR200_INTERFACE_TYPE_SERIAL
    actual_size_transmitted = write(pCtx->fd, pDataOut, dataOutLen);
#else
    #error "Unsupported device interface"
#endif
    if ((uint32_t)actual_size_transmitted != dataOutLen) {
        fprintf(stderr, "Error sending cmd 0x03: dataOutLen=%d, actual_size_transmitted=%d\n", dataOutLen, actual_size_transmitted);
    }
    free(pDataOut);

    ypdr200_frame_prolog_t prologIn = {};
    int actual_size_received = 0;

    #if YPDR200_INTERFACE_TYPE == YPDR200_INTERFACE_TYPE_LIBUSB
    rv = libusb_bulk_transfer(pCtx->handle, YPDR200_BULK_ENDPOINT_ADDR_IN, prologIn.raw, YPDR200_FRAME_PROLOG_SIZE, &actual_size_received, YPDR200_BULK_TRANSFER_TIMEOUT_MS);
    if (rv != 0) {
        fprintf(stderr, "Error reading response prolog: %s (%d)\n", libusb_error_name(rv), rv);
        return YPDR200_X03_ERR_READ_RESPONSE;
    }
    #elif YPDR200_INTERFACE_TYPE == YPDR200_INTERFACE_TYPE_SERIAL
    actual_size_received = read(pCtx->fd, prologIn.raw, YPDR200_FRAME_PROLOG_SIZE);
    #endif
    if (actual_size_received != YPDR200_FRAME_PROLOG_SIZE) {
        fprintf(stderr, "Error reading response prolog: YPD200_FRAME_PROLOG_SIZE=%d, actual_size_received=%d\n", YPDR200_FRAME_PROLOG_SIZE, actual_size_received);
        fprintf(stderr, "The received incomplete prolog is: ");
        for (int i = 0; i < actual_size_received; i++) {
            fprintf(stderr, "0x%02X ", prologIn.raw[i]);
        }
        fprintf(stderr, "\n");
        return YPDR200_X03_ERR_READ_RESPONSE;
    }

    ///<debug print received prolog>
    fprintf(stdout, "Received prolog: ");
    for (int i = 0; i < YPDR200_FRAME_PROLOG_SIZE; i++) {
        fprintf(stdout, "0x%02X ", prologIn.raw[i]);
    }
    fprintf(stdout, "\n");

    uint16_t paramInLen = ypdr200_frame_prolog_get_param_length(&prologIn);
    uint8_t* pParamIn = (uint8_t*)0;
    if (paramInLen > 0) {
        pParamIn = (uint8_t*) malloc(paramInLen);
        if (pParamIn == (uint8_t*)0) {
            errno = ENOMEM;
            return YPDR200_X03_ERR_READ_RESPONSE;
        }
        actual_size_received = 0;
        #if YPDR200_INTERFACE_TYPE == YPDR200_INTERFACE_TYPE_LIBUSB
        rv = libusb_bulk_transfer(pCtx->handle, YPDR200_BULK_ENDPOINT_ADDR_IN, pParamIn, paramInLen, &actual_size_received, YPDR200_BULK_TRANSFER_TIMEOUT_MS);
        if (rv != 0) {
            fprintf(stderr, "Error reading response param data: %s (%d)\n", libusb_error_name(rv), rv);
            free(pParamIn);
            return YPDR200_X03_ERR_READ_RESPONSE;
        }
        #elif YPDR200_INTERFACE_TYPE == YPDR200_INTERFACE_TYPE_SERIAL
        actual_size_received = read(pCtx->fd, pParamIn, paramInLen);
        #endif
        if (actual_size_received != paramInLen) {
            fprintf(stderr, "Error reading response param data: paramInLen=%d, actual_size_received=%d\n", paramInLen, actual_size_received);
            free(pParamIn);
            return YPDR200_X03_ERR_READ_RESPONSE;
        }

        ///<debug print received param data>
        fprintf(stdout, "Received param data: ");
        for (int i = 0; i < paramInLen; i++) {
            fprintf(stdout, "0x%02X ", pParamIn[i]);
        }
        fprintf(stdout, "\n");
        ///</debug print received param data>
    }

    ypdr200_frame_epilog_t epilogIn = {};
    actual_size_received = 0;
    #if YPDR200_INTERFACE_TYPE == YPDR200_INTERFACE_TYPE_LIBUSB
    rv = libusb_bulk_transfer(pCtx->handle, YPDR200_BULK_ENDPOINT_ADDR_IN, epilogIn.raw, YPDR200_FRAME_EPILOG_SIZE, &actual_size_received, YPDR200_BULK_TRANSFER_TIMEOUT_MS);
    if (rv != 0) {
        fprintf(stderr, "Error reading response epilog: %s (%d)\n", libusb_error_name(rv), rv);
        if (pParamIn != (uint8_t*)0) {
            free(pParamIn);
        }
        return YPDR200_X03_ERR_READ_RESPONSE;
    }
    #elif YPDR200_INTERFACE_TYPE == YPDR200_INTERFACE_TYPE_SERIAL
    actual_size_received = read(pCtx->fd, epilogIn.raw, YPDR200_FRAME_EPILOG_SIZE);
    #endif
    if (actual_size_received != YPDR200_FRAME_EPILOG_SIZE) {
        fprintf(stderr, "Error reading response epilog: YPD200_FRAME_EPILOG_SIZE=%d, actual_size_received=%d\n", YPDR200_FRAME_EPILOG_SIZE, actual_size_received);
        if (pParamIn != (uint8_t*)0) {
            free(pParamIn);
        }
        return YPDR200_X03_ERR_READ_RESPONSE;
    }

    ///<debug print received epilog>
    fprintf(stdout, "Received epilog: ");
    for (int i = 0; i < YPDR200_FRAME_EPILOG_SIZE; i++) {
        fprintf(stdout, "0x%02X ", epilogIn.raw[i]);
    }
    fprintf(stdout, "\n");
    ///</debug print received epilog>

    ypdr200_frame_t frameIn = (ypdr200_frame_t) {
        .prolog = prologIn,
        .epilog = epilogIn,
        .pParamData = pParamIn
    };

    /// <Print the received frame for debugging>
    fprintf(stdout, "Received frame: ");
    uint8_t *pRawIn = ypdr200_frame_raw(&frameIn);
    uint32_t frameInLen = ypdr200_frame_length(&frameIn);
    for (uint32_t i = 0; i < frameInLen; i++) {
        fprintf(stdout, "0x%02X ", pRawIn[i]);
    }
    fprintf(stdout, "\n");
    free(pRawIn);
    /// </Print the received frame for debugging>
    /// <Receive any remaining data (for debugging)>
    uint8_t remainingData[1024];
    actual_size_received = 0;
    #if YPDR200_INTERFACE_TYPE == YPDR200_INTERFACE_TYPE_LIBUSB
    rv = libusb_bulk_transfer(pCtx->handle, YPDR200_BULK_ENDPOINT_ADDR_IN, remainingData, sizeof(remainingData), &actual_size_received, YPDR200_BULK_TRANSFER_TIMEOUT_MS);
    if (rv != 0) {
        fprintf(stderr, "Error reading remaining data: %s (%d)\n", libusb_error_name(rv), rv);
        return YPDR200_X03_ERR_READ_RESPONSE;
    }
    #elif YPDR200_INTERFACE_TYPE == YPDR200_INTERFACE_TYPE_SERIAL
    actual_size_received = read(pCtx->fd, remainingData, sizeof(remainingData));
    #endif
    if (actual_size_received > 0) {
        fprintf(stdout, "Received %d bytes of remaining data: ", actual_size_received);
        for (int i = 0; i < actual_size_received; i++) {
            fprintf(stdout, "0x%02X ", remainingData[i]);
        }
        fprintf(stdout, "\n");
    } else {
        fprintf(stdout, "No remaining data received\n");
    }
    /// </Receive any remaining data (for debugging)>

    uint8_t checksum = ypdr200_frame_get_checksum(&frameIn);
    if (checksum != epilogIn.checksum) {
        fprintf(stderr, "Checksum mismatch: expected=0x%02X, actual=0x%02X\n", checksum, epilogIn.checksum);
        /*if (pParamIn != (uint8_t*)0) {
            free(pParamIn);
        }
        return YPDR200_X03_ERR_READ_RESPONSE;*/
        fprintf(stderr, "Ignoring checksum mismatch!\n");
    }

    if (frameIn.prolog.type != YPDR200_FRAME_TYPE_RESPONSE) {
        fprintf(stderr, "Unexpected frame type: expected=0x%02X, actual=0x%02X\n", YPDR200_FRAME_TYPE_RESPONSE, frameIn.prolog.type);
        if (pParamIn != (uint8_t*)0) {
            free(pParamIn);
        }
        return YPDR200_X03_ERR_READ_RESPONSE;
    }

    if (frameIn.prolog.cmd != YPDR200_FRAME_CMD_X03) {
        fprintf(stderr, "Unexpected frame cmd: expected=0x%02X, actual=0x%02X\n", YPDR200_FRAME_CMD_X03, frameIn.prolog.cmd);
        if (pParamIn != (uint8_t*)0) {
            free(pParamIn);
        }
        return YPDR200_X03_ERR_READ_RESPONSE;
    }

    if (pParamIn) {
        *ppcInfo_out = (char*) malloc(paramInLen + 1);
        if (*ppcInfo_out == (char*)0) {
            errno = ENOMEM;
            if (pParamIn != (uint8_t*)0) {
                free(pParamIn);
            }
            return YPDR200_X03_ERR_READ_RESPONSE;
        }
        memcpy(*ppcInfo_out, pParamIn, paramInLen);
        (*ppcInfo_out)[paramInLen] = '\0';

        free(pParamIn);
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