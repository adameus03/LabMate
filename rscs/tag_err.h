#ifndef TAG_ERR_H
#define TAG_ERR_H

/**
 * @file tag_err.h
 * @brief Specification of error codes for tag operations in accordance with EPC Gen2 protocol GS1 standard
 */

#include "uhfman_common.h"

#if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
#include "ypdr200.h"

#define TAG_ERR_ACCESS_DENIED YPDR200_RESP_ERR_CODE_ACCESS_FAIL
#endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200

/* Catch-all for errors not covered by other codes */
#define TAG_GEN2_ERR_OTHER 0x00
/* The Tag does not support the specified parameters or feature */
#define TAG_GEN2_ERR_NOT_SUPPORTED 0x01
/* The Interrogator did not authenticate itself with sufficient privileg-
es for the Tag to perform the operation */
#define TAG_GEN2_ERR_INSUFFICIENT_PRIVILEGES 0x02
/* The Tag memory location does not exist, is too small, or the Tag
does not support the specified EPC length */
#define TAG_GEN2_ERR_MEMORY_OVERRUN 0x03
/* The Tag memory location is locked or permalocked and is either
not writeable or not readable.*/
#define TAG_GEN2_ERR_MEMORY_LOCKED 0x04
/* Catch-all for errors specified by the cryptographic suite */
#define TAG_GEN2_ERR_CRYPTO_SUITE 0x05
/* The Interrogator did not encapsulate the command in an
AuthComm or SecureComm as required */
#define TAG_GEN2_ERR_ENCAPSULATION 0x06
/* The operation failed because the ResponseBuffer overflowed */
#define TAG_GEN2_ERR_RESP_BUFFER_OVERFLOW 0x07
/* The command failed because the Tag is in a security timeout */
#define TAG_GEN2_ERR_SECURITY_TIMEOUT 0x08
/* The Tag has insufficient power to perform the operation */
#define TAG_GEN2_ERR_INSUFFICIENT_POWER 0x0B
/* The Tag does not support error-specific codes */
#define TAG_GEN2_ERR_GENERIC 0x0F

#define TAG_GEN2_ERR_UNKNOWN 0xFF

typedef enum {
    TAG_GEN2_ERR_TYPE_READ = 0x00,
    TAG_GEN2_ERR_TYPE_WRITE = 0x01,
    TAG_GEN2_ERR_TYPE_LOCK = 0x02,
    TAG_GEN2_ERR_TYPE_KILL = 0x03,
    TAG_GEN2_ERR_TYPE_BLOCK_PERMALOCK = 0x04,
    TAG_GEN2_ERR_TYPE_OTHER = 0x05
} tag_gen2_err_type_t;

/**
 * @param rerr Response error code
 * @returns TAG_GEN2_ERR_* error code
 */
uint8_t tag_gen2_err_resolve(uint8_t rerr, tag_gen2_err_type_t* pErrType_out);

#endif // TAG_ERR_H