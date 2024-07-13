#include <stdint.h>

#define QDA_U8BUF2BMP_ERR_SUCCESS 0
#define QDA_U8BUF2BMP_ERR_QRCODE_ENCODE_DATA 1
#define QDA_U8BUF2BMP_ERR_MALLOC 2
int qda_u8buf2bmp(const uint8_t* pInputData, const int nInputDataLen, uint8_t** ppOutputData, int* pnOutputWidth, int* pnOutputHeight);