#include "ypdr200.h"
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// lsusb -d1a86:7523 -v

#define YPDR200_BULK_ENDPOINT_ADDR_IN 0x82U
#define YPDR200_BULK_ENDPOINT_ADDR_OUT 0x02U
#define YPDR200_INTERRUPT_ENDPOINT_ADDR_IN 0x81U

#define YPDR200_BULK_TRANSFER_TIMEOUT_MS 5000

#define YPDR200_FRAME_PROLOG_SIZE 5
typedef union _ypdr200_frame_prolog {
    struct {
        const uint8_t header;
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
        const uint8_t end;
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
    uint8_t* pRaw = (uint8_t*) malloc((size_t)paramLen);
    if (pRaw == (uint8_t*)0) {
        errno = ENOMEM;
        return (uint8_t*)0;
    }
    uint8_t* pBlk = pRaw;
    memcpy(pBlk, prolog.raw, sizeof(prolog.raw));
    pBlk += sizeof(prolog.raw);
    memcpy(pBlk, pFrame->pParamData, (size_t)paramLen);
    pBlk += (size_t)paramLen;
    memcpy(pBlk, pFrame->epilog.raw, sizeof(pFrame->epilog.raw));
    return pRaw;
}

int ypdr200_x03(uhfman_ctx_t* pCtx, ypdr200_x03_param_t infoType, char** ppcInfo_out) {
    //TODO implement
    return -1;
}