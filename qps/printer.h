#include <stdint.h>
#include "libusb.h"

#define PRINTER_USE_DEBUG_EXTENSIONS 1
#define PRINTER_USB_IFACE_IX 0
//#define PRINTER_USB_IFACE_IX 50

#define PRINTER_ERR_SUCCESS 0
#define PRINTER_ERR_LIBUSB_INIT 1
#define PRINTER_ERR_DEVICE_NOT_FOUND 2
#define PRINTER_ERR_INTERFACE_CLAIM 3
#define PRINTER_ERR_READ_RESPONSE 4
#define PRINTER_ERR_SEND_COMMAND 5
#define PRINTER_ERR_INTERFACE_RELEASE 6

typedef struct {
    libusb_device_handle *handle;
    libusb_context *context;
} printer_ctx_t;

#define PRINTER_ESC_V_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define PRINTER_ESC_V_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
#define PRINTER_ESC_V_ERR_READ_RESPONSE PRINTER_ERR_READ_RESPONSE
/**
 * @brief Request print engine version
 * TODO: Don't print to stdout, but return response via a pointer. Or keep it? for additional layer of logs...
 */
int printer_esc_v(printer_ctx_t* pCtx);

#define PRINTER_ESC_D_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define PRINTER_ESC_D_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * @brief 
 */
int printer_esc_d(printer_ctx_t* pCtx, uint8_t *data, int width, int height);

#define PRINTER_TAKE_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define PRINTER_TAKE_ERR_LIBUSB_INIT PRINTER_ERR_LIBUSB_INIT
#define PRINTER_TAKE_ERR_DEVICE_NOT_FOUND PRINTER_ERR_DEVICE_NOT_FOUND
#define PRINTER_TAKE_ERR_INTERFACE_CLAIM PRINTER_ERR_INTERFACE_CLAIM
/**
 * @brief Initializes underlying USB library and attempts to find the printer and claim the interface
 */
int printer_take(printer_ctx_t *pCtx_out);

/**
 * @brief Releases the interface and closes the device
 */
void printer_release(printer_ctx_t *pCtx);
