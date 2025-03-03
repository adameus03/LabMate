#include "tag_err.h"

#if UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200
uint8_t tag_gen2_err_resolve(uint8_t rerr, tag_gen2_err_type_t* pErrType_out) {
    switch (rerr & 0xF0) {
        case YPDR200_RESP_ERR_CODE_READ_ERROR_MASK:
            if (pErrType_out) *pErrType_out = TAG_GEN2_ERR_TYPE_READ;
            break;
        case YPDR200_RESP_ERR_CODE_WRITE_ERROR_MASK:
            if (pErrType_out) *pErrType_out = TAG_GEN2_ERR_TYPE_WRITE;
            break;
        case YPDR200_RESP_ERR_CODE_LOCK_ERROR_MASK:
            if (pErrType_out) *pErrType_out = TAG_GEN2_ERR_TYPE_LOCK;
            break;
        case YPDR200_RESP_ERR_CODE_KILL_ERROR_MASK:
            if (pErrType_out) *pErrType_out = TAG_GEN2_ERR_TYPE_KILL;
            break;
        case YPDR200_RESP_ERR_CODE_BLOCK_PERMALOCK_ERROR_MASK:
            if (pErrType_out) *pErrType_out = TAG_GEN2_ERR_TYPE_BLOCK_PERMALOCK;
            break;
        default:        
            switch (rerr) {
                case YPDR200_RESP_ERR_CODE_READ_FAIL:
                    if (pErrType_out) *pErrType_out = TAG_GEN2_ERR_TYPE_READ;
                    break;
                case YPDR200_RESP_ERR_CODE_WRITE_FAIL:
                    if (pErrType_out) *pErrType_out = TAG_GEN2_ERR_TYPE_WRITE;
                    break;
                case YPDR200_RESP_ERR_CODE_LOCK_FAIL:
                    if (pErrType_out) *pErrType_out = TAG_GEN2_ERR_TYPE_LOCK;
                    break;
                case YPDR200_RESP_ERR_CODE_KILL_FAIL:
                    if (pErrType_out) *pErrType_out = TAG_GEN2_ERR_TYPE_KILL;
                    break;
                case YPDR200_RESP_ERR_CODE_BLOCK_PERMALOCK_FAIL:
                    if (pErrType_out) *pErrType_out = TAG_GEN2_ERR_TYPE_BLOCK_PERMALOCK;
                    break;
                default:
                    if (pErrType_out) *pErrType_out = TAG_GEN2_ERR_TYPE_OTHER;
                    break;
            }
            return TAG_GEN2_ERR_UNKNOWN;
    }
    return rerr & 0x0F;
}
#endif // UHFMAN_DEVICE_MODEL == UHFMAN_DEVICE_MODEL_YDPR200