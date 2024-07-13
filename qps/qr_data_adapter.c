#include "qrencode.h"
#include <stdlib.h>
#include <stdio.h> // For debugging purposes

#include "qr_data_adapter.h"

#pragma pack(push, 1)
typedef struct {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
} BITMAPFILEHEADER;
typedef struct {
    uint32_t biSize;
    int32_t biWidth;
    int32_t biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t biXPelsPerMeter;
    int32_t biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
} BITMAPINFOHEADER;
#pragma pack(pop)


int qda_dlw500u8buf_to_grayscale(const uint8_t* pInputData, const int nInputDataLen, uint8_t** ppOutputData, int* pnOutputWidth, int* pnOutputHeight) {
    QRcode* pQRcode = QRcode_encodeData(nInputDataLen, pInputData, 0, QR_ECLEVEL_L);
    if (pQRcode == (QRcode*)0) {
        *ppOutputData = (uint8_t*)0;
        *pnOutputWidth = 0;
        *pnOutputHeight = 0;
        return QDA_U8BUF2BMP_ERR_QRCODE_ENCODE_DATA;
    }

    // The output bitmap is greyscale, so we only need one byte per pixel. Pixel is either black or white depending on LSB of bytes in pQRcode->data.
    int nOutputDataLen = pQRcode->width * pQRcode->width;
    uint8_t* pOutputData = (uint8_t*)malloc(nOutputDataLen);
    if (pOutputData == (uint8_t*)0) {
        QRcode_free(pQRcode);
        *ppOutputData = (uint8_t*)0;
        *pnOutputWidth = 0;
        *pnOutputHeight = 0;
        return QDA_U8BUF2BMP_ERR_MALLOC;
    }

    for (int i = 0; i < nOutputDataLen; i++) {
        pOutputData[i] = (pQRcode->data[i] & 0x01U) ? 0x00U : 0xFFU;
    }

    *ppOutputData = pOutputData;
    *pnOutputWidth = pQRcode->width;
    *pnOutputHeight = pQRcode->width;

    QRcode_free(pQRcode);
    return QDA_U8BUF2BMP_ERR_SUCCESS;
}

int qda_grayscale_expand_pixels(const uint8_t* pInputData, const int nInputWidth, const int nInputHeight, uint8_t** ppOutputData, int* pnOutputWidth, int* pnOutputHeight, const int nPxPerDotDim) {
    int nOutputWidth = nInputWidth * nPxPerDotDim;
    int nOutputHeight = nInputHeight * nPxPerDotDim;
    int nOutputDataLen = nOutputWidth * nOutputHeight;

    uint8_t* pOutputData = (uint8_t*)malloc(nOutputDataLen);
    if (pOutputData == (uint8_t*)0) {
        *ppOutputData = (uint8_t*)0;
        *pnOutputWidth = 0;
        *pnOutputHeight = 0;
        return QDA_GRAYSCALE_EXPAND_PIXELS_ERR_MALLOC;
    }

    for (int y = 0; y < nInputHeight; y++) {
        for (int x = 0; x < nInputWidth; x++) {
            for (int i = 0; i < nPxPerDotDim; i++) {
                for (int j = 0; j < nPxPerDotDim; j++) {
                    pOutputData[(y * nPxPerDotDim + i) * nOutputWidth + x * nPxPerDotDim + j] = pInputData[y * nInputWidth + x];
                }
            }
        }
    }

    *ppOutputData = pOutputData;
    *pnOutputWidth = nOutputWidth;
    *pnOutputHeight = nOutputHeight;

    return QDA_GRAYSCALE_EXPAND_PIXELS_ERR_SUCCESS;
}

int qda_grayscale_pad(qda_grayscale_pad_mode mode, const uint8_t* pInputData, const int nInputWidth, const int nInputHeight, uint8_t** ppOutputData, int* pnOutputWidth, int* pnOutputHeight, const int nPxPadding) {
    switch (mode) {
        case QDA_GRAYSCALE_PAD_MODE_ALL_SIDES:
            int nOutputWidth = nInputWidth + 2 * nPxPadding;
            int nOutputHeight = nInputHeight + 2 * nPxPadding;
            uint8_t* pPaddedData = (uint8_t*)malloc(nOutputWidth * nOutputHeight);
            if (pPaddedData == (uint8_t*)0) {
                *ppOutputData = (uint8_t*)0;
                *pnOutputWidth = 0;
                *pnOutputHeight = 0;
                return QDA_GRAYSCALE_PAD_ERR_MALLOC;
            }

            for (int i = 0; i < nInputHeight; i++) {
                for (int j = 0; j < nInputWidth; j++) {
                    pPaddedData[(i + nPxPadding) * nOutputWidth + j + nPxPadding] = pInputData[i * nInputWidth + j];
                }
            }

            for (int i = 0; i < nPxPadding; i++) {
                for (int j = 0; j < nOutputWidth; j++) {
                    pPaddedData[i * nOutputWidth + j] = 0xFF;
                    pPaddedData[(nOutputHeight - 1 - i) * nOutputWidth + j] = 0xFF;
                }
            }

            for (int i = 0; i < nOutputHeight; i++) {
                for (int j = 0; j < nPxPadding; j++) {
                    pPaddedData[i * nOutputWidth + j] = 0xFF;
                    pPaddedData[i * nOutputWidth + nOutputWidth - 1 - j] = 0xFF;
                }
            }

            *ppOutputData = pPaddedData;
            *pnOutputWidth = nOutputWidth;
            *pnOutputHeight = nOutputHeight;

            return QDA_GRAYSCALE_PAD_ERR_SUCCESS;
        default:
            return QDA_GRAYSCALE_PAD_ERR_INVALID_MODE;
    }
}

int qda_grayscale_to_rgb(const uint8_t* pInputData, const int nWidth, const int nHeight, uint8_t** ppOutputData, int* pnOutputLen) {
    int nInputLen = nWidth * nHeight;
    int nOutputLen = nInputLen * 3;
    uint8_t* pRgbData = (uint8_t*)malloc(nOutputLen);
    if (pRgbData == (uint8_t*)0) {
        *ppOutputData = (uint8_t*)0;
        *pnOutputLen = 0;
        return QDA_GRAYSCALE_TO_RGB_ERR_MALLOC;
    }

    for (int i = 0; i < nInputLen; i++) {
        pRgbData[3 * i + 0] = pInputData[i];
        pRgbData[3 * i + 1] = pInputData[i];
        pRgbData[3 * i + 2] = pInputData[i];
    }

    *ppOutputData = pRgbData;
    *pnOutputLen = nOutputLen;
    return QDA_GRAYSCALE_TO_RGB_ERR_SUCCESS;
}

#if QR_DATA_ADAPTER_USE_DEBUG_EXTENSIONS == 1
void qda_grayscale_print_to_console(const uint8_t* pInputData, const int nWidth, const int nHeight) {
    // For debugging purposes, print the greyscale bitmap to the console.
    for (int y = 0; y < nHeight; y++) {
        for (int x = 0; x < nWidth; x++) {
            printf("%c", pInputData[y * nWidth + x] < 0x80U ? '#' : ' ');
        }
        printf("\n");
    }
}

int qda_rgb_save_to_bmp_file(const uint8_t* pInputData, const int nWidth, const int nHeight, const int nChannels, const char* pFilePath) {
    // For debugging purposes, write the this greyscale bitmap to a file.
    fprintf(stdout, "Converting and writing QR code to file %s\n", pFilePath);
    fprintf(stdout, "Absolute containing directory path: %s\n", __FILE__);

    int nInputLen = nWidth * nHeight * nChannels;

    BITMAPFILEHEADER fileHeader = {
        .bfType = 0x4D42,
        .bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + nInputLen,
        .bfReserved1 = 0,
        .bfReserved2 = 0,
        .bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER)
    };

    BITMAPINFOHEADER infoHeader = {
        .biSize = sizeof(BITMAPINFOHEADER),
        .biWidth = nWidth,
        .biHeight = nHeight,
        .biPlanes = 1,
        .biBitCount = nChannels * 8,
        .biCompression = 0,
        .biSizeImage = 0,
        .biXPelsPerMeter = 0,
        .biYPelsPerMeter = 0,
        .biClrUsed = 0,
        .biClrImportant = 0
    };

    FILE* pFile = fopen(pFilePath, "wb");
    if (pFile != (FILE*)0) {
        fwrite(&fileHeader, sizeof(BITMAPFILEHEADER), 1, pFile);
        fwrite(&infoHeader, sizeof(BITMAPINFOHEADER), 1, pFile);
        fwrite(pInputData, nInputLen, 1, pFile);
        fclose(pFile);
        fprintf(stdout, "Finished writing QR code to file\n");
        return QDA_RGB_SAVE_TO_BMP_FILE_ERR_SUCCESS;
    } else {
        return QDA_RGB_SAVE_TO_BMP_FILE_ERR_FILE_OPEN_FAILED;
    }
}
#endif