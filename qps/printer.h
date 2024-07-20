#include <stdint.h>
#include "printer_common.h"

#define PRINTER_MODEL_DYMO_LABELWRITER_400 1
#define PRINTER_MODEL PRINTER_MODEL_DYMO_LABELWRITER_400

#if PRINTER_MODEL == PRINTER_MODEL_DYMO_LABELWRITER_400
    #define PRINTER_VENDOR_ID 0x0922
    #define PRINTER_PRODUCT_ID 0x0019
    #define PRINTER_USB_IFACE_IX 0
    #define PRINTER_REVISION_STRING_LENGTH 8
    #define PRINTER_REVISION_STRING_SUPPORTED
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
