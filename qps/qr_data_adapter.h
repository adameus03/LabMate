#include <stdint.h>

#define QR_DATA_ADAPTER_USE_DEBUG_EXTENSIONS 1

typedef enum {
    QDA_GRAYSCALE_PAD_MODE_ALL_SIDES = 0
} qda_grayscale_pad_mode;

#define QDA_U8BUF2BMP_ERR_SUCCESS 0
#define QDA_U8BUF2BMP_ERR_QRCODE_ENCODE_DATA 1
#define QDA_U8BUF2BMP_ERR_MALLOC 2
int qda_dlw500u8buf_to_grayscale(const uint8_t* pInputData, const int nInputDataLen, uint8_t** ppOutputData, int* pnOutputWidth, int* pnOutputHeight);

#define QDA_GRAYSCALE_EXPAND_PIXELS_ERR_SUCCESS 0
#define QDA_GRAYSCALE_EXPAND_PIXELS_ERR_MALLOC 1
int qda_grayscale_expand_pixels(const uint8_t* pInputData, const int nInputWidth, const int nInputHeight, uint8_t** ppOutputData, int* pnOutputWidth, int* pnOutputHeight, const int nPxPerDotDim);

#define QDA_GRAYSCALE_PAD_ERR_SUCCESS 0
#define QDA_GRAYSCALE_PAD_ERR_MALLOC 1
#define QDA_GRAYSCALE_PAD_ERR_INVALID_MODE 2
int qda_grayscale_pad(qda_grayscale_pad_mode mode, const uint8_t* pInputData, const int nInputWidth, const int nInputHeight, uint8_t** ppOutputData, int* pnOutputWidth, int* pnOutputHeight, const int nPxPadding);

#define QDA_GRAYSCALE_TO_RGB_ERR_SUCCESS 0
#define QDA_GRAYSCALE_TO_RGB_ERR_MALLOC 1
int qda_grayscale_to_rgb(const uint8_t* pInputData, const int nWidth, const int nHeight, uint8_t** ppOutputData, int* pnOutputLen);

#if QR_DATA_ADAPTER_USE_DEBUG_EXTENSIONS == 1

void qda_grayscale_print_to_console(const uint8_t* pInputData, const int nWidth, const int nHeight);

#define QDA_RGB_SAVE_TO_BMP_FILE_ERR_SUCCESS 0
#define QDA_RGB_SAVE_TO_BMP_FILE_ERR_FILE_OPEN_FAILED 1
int qda_rgb_save_to_bmp_file(const uint8_t* pInputData, const int nWidth, const int nHeight, const int nChannels, const char* pFilePath);

#endif
