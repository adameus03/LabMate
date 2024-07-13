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


int qda_u8buf2bmp(const uint8_t* pInputData, const int nInputDataLen, uint8_t** ppOutputData, int* pnOutputWidth, int* pnOutputHeight) {
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


    // For debugging purposes, print the greyscale bitmap to the console.
    for (int y = 0; y < pQRcode->width; y++) {
        for (int x = 0; x < pQRcode->width; x++) {
            printf("%c", pOutputData[y * pQRcode->width + x] == 0x00U ? '#' : ' ');
        }
        printf("\n");
    }

    // For debugging purposes, write the this greyscale bitmap to a file.
    fprintf(stdout, "Converting and writing QR code to file\n");
    fprintf(stdout, "Absolute path: %s\n", __FILE__);
    int pxPerDotDim = 8;
    int pxPadding = 8;
    BITMAPFILEHEADER fileHeader = {
        .bfType = 0x4D42,
        .bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + (pQRcode->width * pxPerDotDim + 2 * pxPadding) * (pQRcode->width * pxPerDotDim + 2 * pxPadding) * 3,
        .bfReserved1 = 0,
        .bfReserved2 = 0,
        .bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER)
    };

    BITMAPINFOHEADER infoHeader = {
        .biSize = sizeof(BITMAPINFOHEADER),
        .biWidth = pQRcode->width * pxPerDotDim + 2 * pxPadding,
        .biHeight = pQRcode->width * pxPerDotDim + 2 * pxPadding,
        .biPlanes = 1,
        .biBitCount = 24,
        .biCompression = 0,
        .biSizeImage = 0,
        .biXPelsPerMeter = 0,
        .biYPelsPerMeter = 0,
        .biClrUsed = 0,
        .biClrImportant = 0
    };

    FILE* pFile = fopen("qrcode.bmp", "wb");
    if (pFile != (FILE*)0) {
        fwrite(&fileHeader, sizeof(BITMAPFILEHEADER), 1, pFile);
        fwrite(&infoHeader, sizeof(BITMAPINFOHEADER), 1, pFile);
        
        uint8_t* pPreRgbData = (uint8_t*)malloc(nOutputDataLen * pxPerDotDim * pxPerDotDim);
        for (int y = 0; y < pQRcode->width; y++) {
            for (int x = 0; x < pQRcode->width; x++) {
                for (int i = 0; i < pxPerDotDim; i++) {
                    for (int j = 0; j < pxPerDotDim; j++) {
                        pPreRgbData[(y * pxPerDotDim + i) * pQRcode->width * pxPerDotDim + x * pxPerDotDim + j] = pOutputData[y * pQRcode->width + x];
                        //pPreRgbData[(y * 2 + i) * pQRcode->width * 2 + x * 2 + j] = pOutputData[y * pQRcode->width + x];
                    }
                }

                // uint8_t* pSmallPixel = pOutputData + y * pQRcode->width + x;
                // uint8_t* pBigPixel = pPreRgbData + y * pxPerDotDim * pQRcode->width * pxPerDotDim + x * pxPerDotDim;
                // uint8_t* pBigPixelFirstRow = pBigPixel;
                // uint8_t* pBigPixelSecondRow = pBigPixel + pQRcode->width * pxPerDotDim;
                // pBigPixelFirstRow[0] = pSmallPixel[0];
                // pBigPixelFirstRow[1] = pSmallPixel[0];
                // pBigPixelSecondRow[0] = pSmallPixel[0];
                // pBigPixelSecondRow[1] = pSmallPixel[0];
            }
        }

        uint8_t* pPaddedPreRgbData = (uint8_t*)malloc((pQRcode->width * pxPerDotDim + 2 * pxPadding) * (pQRcode->width * pxPerDotDim + 2 * pxPadding));
        for (int i = 0; i < pQRcode->width * pxPerDotDim; i++) {
            for (int j = 0; j < pQRcode->width * pxPerDotDim; j++) {
                pPaddedPreRgbData[(i + pxPadding) * (pQRcode->width * pxPerDotDim + 2 * pxPadding) + j + pxPadding] = pPreRgbData[i * pQRcode->width * pxPerDotDim + j];
                //pPaddedPreRgbData[(i + )]
            }
        }

        for (int i = 0; i < pxPadding; i++) {
            for (int j = 0; j < pQRcode->width * pxPerDotDim + 2 * pxPadding; j++) {
                pPaddedPreRgbData[i * (pQRcode->width * pxPerDotDim + 2 * pxPadding) + j] = 0xFF;
                pPaddedPreRgbData[(pQRcode->width * pxPerDotDim + 2 * pxPadding - 1 - i) * (pQRcode->width * pxPerDotDim + 2 * pxPadding) + j] = 0xFF;
            }
        }

        for (int i = 0; i < pQRcode->width * pxPerDotDim + 2 * pxPadding; i++) {
            for (int j = 0; j < pxPadding; j++) {
                pPaddedPreRgbData[i * (pQRcode->width * pxPerDotDim + 2 * pxPadding) + j] = 0xFF;
                pPaddedPreRgbData[i * (pQRcode->width * pxPerDotDim + 2 * pxPadding) + pQRcode->width * pxPerDotDim + 2 * pxPadding - 1 - j] = 0xFF;
            }
        }

        uint8_t* pRgbData = (uint8_t*)malloc((pQRcode->width * pxPerDotDim + 2 * pxPadding) * (pQRcode->width * pxPerDotDim + 2 * pxPadding) * 3);

        for (int i = 0; i < (pQRcode->width * pxPerDotDim + 2 * pxPadding) * (pQRcode->width * pxPerDotDim + 2 * pxPadding); i++) {
            pRgbData[3 * i + 0] = pPaddedPreRgbData[i];
            pRgbData[3 * i + 1] = pPaddedPreRgbData[i];
            pRgbData[3 * i + 2] = pPaddedPreRgbData[i];
        }

        fwrite(pRgbData, (pQRcode->width * pxPerDotDim + 2 * pxPadding) * (pQRcode->width * pxPerDotDim + 2 * pxPadding) * 3, 1, pFile);

        // uint8_t* pRgbData = (uint8_t*)malloc(nOutputDataLen * 3);
        // for (int i = 0; i < nOutputDataLen; i++) {
        //     pRgbData[3 * i + 0] = pOutputData[i];
        //     pRgbData[3 * i + 1] = pOutputData[i];
        //     pRgbData[3 * i + 2] = pOutputData[i];
        // }

        // fwrite(pRgbData, nOutputDataLen * 3, 1, pFile);

        fclose(pFile);
        fprintf(stdout, "Finished writing QR code to file\n");

        free(pPreRgbData);
        free(pPaddedPreRgbData);
        free(pRgbData);
    }

    QRcode_free(pQRcode);
    return QDA_U8BUF2BMP_ERR_SUCCESS;
}
