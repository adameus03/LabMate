#include <stdint.h>

#define QR_DATA_ADAPTER_USE_DEBUG_EXTENSIONS 1

#define QDA_ERR_SUCCESS 0
#define QDA_ERR_MALLOC 1
#define QDA_ERR_QRCODE_ENCODE_DATA 2
#define QDA_ERR_INVALID_MODE 3
#define QDA_ERR_FILE_OPEN_FAILED 4

typedef enum {
    QDA_GRAYSCALE_PAD_MODE_ALL_SIDES = 0
} qda_grayscale_pad_mode;

#define QDA_QRENCU8BUF2BMP_ERR_SUCCESS QDA_ERR_SUCCESS
#define QDA_QRENCU8BUF2BMP_ERR_MALLOC QDA_ERR_MALLOC
#define QDA_QRENCU8BUF2BMP_ERR_QRCODE_ENCODE_DATA QDA_ERR_QRCODE_ENCODE_DATA
int qda_qrencu8buf_to_grayscale(const uint8_t* pInputData, const int nInputDataLen, uint8_t** ppOutputData, int* pnOutputWidth, int* pnOutputHeight);

#define QDA_GRAYSCALE_EXPAND_PIXELS_ERR_SUCCESS QDA_ERR_SUCCESS
#define QDA_GRAYSCALE_EXPAND_PIXELS_ERR_MALLOC QDA_ERR_MALLOC
int qda_grayscale_expand_pixels(const uint8_t* pInputData, const int nInputWidth, const int nInputHeight, uint8_t** ppOutputData, int* pnOutputWidth, int* pnOutputHeight, const int nPxPerDotDim);

#define QDA_GRAYSCALE_PAD_ERR_SUCCESS QDA_ERR_SUCCESS
#define QDA_GRAYSCALE_PAD_ERR_MALLOC QDA_ERR_MALLOC
#define QDA_GRAYSCALE_PAD_ERR_INVALID_MODE QDA_ERR_INVALID_MODE
int qda_grayscale_pad(qda_grayscale_pad_mode mode, const uint8_t* pInputData, const int nInputWidth, const int nInputHeight, uint8_t** ppOutputData, int* pnOutputWidth, int* pnOutputHeight, const int nPxPadding);

#define QDA_GRAYSCALE_TO_RGB_ERR_SUCCESS QDA_ERR_SUCCESS
#define QDA_GRAYSCALE_TO_RGB_ERR_MALLOC QDA_ERR_MALLOC
int qda_grayscale_to_rgb(const uint8_t* pInputData, const int nWidth, const int nHeight, uint8_t** ppOutputData, int* pnOutputLen);

#if QR_DATA_ADAPTER_USE_DEBUG_EXTENSIONS == 1
void qda_grayscale_print_to_console(const uint8_t* pInputData, const int nWidth, const int nHeight);

#define QDA_RGB_SAVE_TO_BMP_FILE_ERR_SUCCESS QDA_ERR_SUCCESS
#define QDA_RGB_SAVE_TO_BMP_FILE_ERR_FILE_OPEN_FAILED QDA_ERR_FILE_OPEN_FAILED
int qda_rgb_save_to_bmp_file(const uint8_t* pInputData, const int nWidth, const int nHeight, const int nChannels, const char* pFilePath);
#endif // QR_DATA_ADAPTER_USE_DEBUG_EXTENSIONS == 1

#define QDA_GRAYSCALE_TO_DLW400U8BUF_ERR_SUCCESS QDA_ERR_SUCCESS
#define QDA_GRAYSCALE_TO_DLW400U8BUF_ERR_MALLOC QDA_ERR_MALLOC
/**
 * @param pnOutputLen Number of bytes in the output buffer. The buffer consists of data lines, each line is nWidth / 8 bytes long.
 */
int qda_grayscale_to_dlw400u8buf(const uint8_t* pInputData, const int nWidth, const int nHeight, uint8_t** ppOutputData, int* pnOutputLen);

#define QDA_GRAYSCALE_DIPTYCH_TO_DLW400U8BUF_ERR_SUCCESS QDA_ERR_SUCCESS
#define QDA_GRAYSCALE_DIPTYCH_TO_DLW400U8BUF_ERR_MALLOC QDA_ERR_MALLOC
/**
 * @brief Utility function for converting a grayscale diptych to a single DLW400U8BUF buffer. It is particulary useful for printing two QR codes on a label which consists of two separate tearable parts aligned horizontally (diptych).
 * @param pInputDataDiptychLeft Pointer to the grayscale diptych data for the left part of the diptych.
 * @param pInputDataDiptychRight Pointer to the grayscale diptych data for the right part of the diptych.
 * @param nWidth Width of each part of the 2 parts of the diptych.
 * @param nHeight Height of each part of the 2 parts of the diptych.
 * @param nSeparationWidth Width of the separation between the 2 parts of the diptych.
 * @param ppOutputData Pointer to the output buffer containing the DLW400U8BUF data which can be used for printing on a single 11353 label.
 * 
 */
int qda_grayscale_diptych_to_dlw400u8buf(const uint8_t* pInputDataDiptychLeft, const uint8_t* pInputDataDiptychRight, const int nWidth, const int nHeight, const int nSeparationWidth, uint8_t** ppOutputData, int* pnOutputLen);