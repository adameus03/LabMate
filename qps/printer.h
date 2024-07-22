#include <stdint.h>
#include "printer_common.h"

#define PRINTER_MODEL_DYMO_LABELWRITER_400 1
#define PRINTER_MODEL PRINTER_MODEL_DYMO_LABELWRITER_400

#define PRINTER_LABEL_TYPE_FOR_EPPENDORFS 1
/* 11353 Dymo labels - Rozmiar dla eppendorfów */
#define PRINTER_LABEL_TYPE PRINTER_LABEL_TYPE_FOR_EPPENDORFS

#if PRINTER_LABEL_TYPE == PRINTER_LABEL_TYPE_FOR_EPPENDORFS
    #define PRINTER_LABEL_WIDTH_MM 13
    #define PRINTER_LABEL_HEIGHT_MM 25
    //#define PRINTER_LABEL_NUM_SUBLABELS_HORIZONTAL 2
    //#define PRINTER_LABEL_NUM_SUBLABELS_VERTICAL 1
    #define PRINTER_LABEL_IS_SUBDIVIDED 1
    //#define PRINTER_LABEL_MARGIN_LEFT_UM 2000
    #define PRINTER_LABEL_MARGIN_LEFT_UM 0
#else
#error "Unknown label type"
#endif

#if PRINTER_MODEL == PRINTER_MODEL_DYMO_LABELWRITER_400
    #define PRINTER_VENDOR_ID 0x0922
    #define PRINTER_PRODUCT_ID 0x0019
    #define PRINTER_USB_IFACE_IX 0
    #define PRINTER_REVISION_STRING_LENGTH 8
    #define PRINTER_REVISION_STRING_SUPPORTED
    #define PRINTER_MAX_LABEL_WIDTH_MM 57
#else
#error "Unknown printer model"
#endif



#define PRINTER_TAKE_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define PRINTER_TAKE_ERR_LIBUSB_INIT PRINTER_ERR_LIBUSB_INIT
#define PRINTER_TAKE_ERR_DEVICE_NOT_FOUND PRINTER_ERR_DEVICE_NOT_FOUND
#define PRINTER_TAKE_ERR_INTERFACE_CLAIM PRINTER_ERR_INTERFACE_CLAIM
/**
 * @brief Initializes underlying USB library and attempts to find the printer and claim the interface
 */
printer_err_t printer_take(printer_ctx_t *pCtx_out);

/**
 * @brief Releases the interface and closes the device
 */
void printer_release(printer_ctx_t *pCtx);

// #if PRINTER_MODEL == DYMO_LABELWRITER_400
// typedef union {
//     uint8_t raw[8];
//     struct {
//         char model_number[4];
//         char lowercase_letter[1];
//         char firmware_version[2];
//     };
// } printer_dymo_lw400_revision_t;
// #endif

#define PRINTER_GET_REVISION_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define PRINTER_GET_REVISION_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
#define PRINTER_GET_REVISION_ERR_READ_RESPONSE PRINTER_ERR_READ_RESPONSE
#define PRINTER_GET_REVISION_ERR_NOT_SUPPORTED PRINTER_ERR_NOT_SUPPORTED
#define PRINTER_GET_REVISION_ERR_UNKNOWN_PRINTER_MODEL PRINTER_ERR_UNKNOWN_PRINTER_MODEL
/**
 * @brief Obtains the printer's revision string if supported
 * @param pcRevision_out pointer to output string - you should allocate PRINTER_REVISION_STRING_LENGTH+1 bytes (PRINTER_REVISION_STRING_LENGTH for the revision and 1 for the null terminator)
 */
printer_err_t printer_get_revision(printer_ctx_t *pCtx, char *pcRevision_out);

// TODO: For now we are assuming no label subdivision (printing on the first sublabel only), but this should be fixed later after no-subdivision testing.
// TODO: Convert PRINTER_LABEL_TYPE to a function parameter?
// TODO: Detect the printer model automatically using its VID+PID and a list of supported printer models

#define PRINTER_SETUP_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define PRINTER_SETUP_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
#define PRINTER_SETUP_ERR_READ_RESPONSE PRINTER_ERR_READ_RESPONSE
#define PRINTER_SETUP_ERR_UNKNOWN_PRINTER_MODEL PRINTER_ERR_UNKNOWN_PRINTER_MODEL
/**
 * @brief Configures the printer settings depending on PRINTER_LABEL_TYPE
 */
printer_err_t printer_setup(printer_ctx_t *pCtx);

#define PRINTER_PRINT_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define PRINTER_PRINT_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
#define PRINTER_PRINT_ERR_READ_RESPONSE PRINTER_ERR_READ_RESPONSE
#define PRINTER_PRINT_ERR_CONVERSION_FAILED PRINTER_ERR_CONVERSION_FAILED
//#define PRINTER_PRINT_ERR_MALLOC PRINTER_ERR_MALLOC
#define PRINTER_PRINT_ERR_UNKNOWN_PRINTER_MODEL PRINTER_ERR_UNKNOWN_PRINTER_MODEL

#if PRINTER_LABEL_IS_SUBDIVIDED == 0
/**
 * @brief Prints a label
 * @param pLabelGrayscaleData pointer to the label grayscale data
 * @param labelGrayscaleDataWidth width of the label grayscale data
 * @param labelGrayscaleDataHeight height of the label grayscale data
 * 
 * @note labelGrayscaleDataWidth and labelGrayscaleDataHeight can be smaller than (physical) label width and height respectively
 */
printer_err_t printer_print(printer_ctx_t *pCtx, const uint8_t* pLabelGrayscaleData, const int labelGrayscaleDataWidth, const int labelGrayscaleDataHeight);
#else
/**
 * @brief Prints a label
 * @param pLabelGrayscaleDataDiptychLeft pointer to the label grayscale data for the left part of the diptych
 * @param pLabelGrayscaleDataDiptychRight pointer to the label grayscale data for the right part of the diptych
 * @param labelGrayscaleDataWidth width of the label grayscale data for each part of the diptych
 * @param labelGrayscaleDataHeight height of the label grayscale data for each part of the diptych
 * 
 * @note labelGrayscaleDataWidth and labelGrayscaleDataHeight can be smaller than (physical) label width and height respectively
 */
printer_err_t printer_print(printer_ctx_t *pCtx, const uint8_t* pLabelGrayscaleDataDiptychLeft, const uint8_t* pLabelGrayscaleDataDiptychRight, const int labelGrayscaleDataWidth, const int labelGrayscaleDataHeight);
#endif //PRINTER_LABEL_IS_SUBDIVIDED == 0



