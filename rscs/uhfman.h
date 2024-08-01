/**
 * @file uhfman.h
 * @brief UHF RFID reader/writer manager
 */

#include "uhfman_common.h"

#define UHFMAN_TAKE_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_TAKE_ERR_LIBUSB_INIT UHFMAN_ERR_LIBUSB_INIT
#define UHFMAN_TAKE_ERR_DEVICE_NOT_FOUND UHFMAN_ERR_DEVICE_NOT_FOUND
#define UHFMAN_TAKE_ERR_INTERFACE_CLAIM UHFMAN_ERR_INTERFACE_CLAIM
#define UHFMAN_TAKE_ERR_BRIDGE_INIT_FAIL UHFMAN_ERR_BRIDGE_INIT_FAIL
/**
 * @brief Initializes underlying USB library and attempts to find the UHF RFID reader/writer and claim the interface
 */
uhfman_err_t uhfman_device_take(uhfman_ctx_t *pCtx_out);

/**
 * @brief Releases the interface and closes the device
 */
void uhfman_device_release(uhfman_ctx_t *pCtx);

#define UHFMAN_GET_HARDWARE_VERSION_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_GET_HARDWARE_VERSION_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_GET_HARDWARE_VERSION_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_GET_HARDWARE_VERSION_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_GET_HARDWARE_VERSION_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_GET_HARDWARE_VERSION_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_GET_HARDWARE_VERSION_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
/**
 * @brief Get hardware version of the reader/writer module
 * @param ppVersion_out Address where the output version string pointer will be stored
 */
uhfman_err_t uhfman_get_hardware_version(uhfman_ctx_t* pCtx, char** ppcVersion_out);

#define UHFMAN_GET_SOFTWARE_VERSION_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_GET_SOFTWARE_VERSION_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_GET_SOFTWARE_VERSION_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_GET_SOFTWARE_VERSION_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_GET_SOFTWARE_VERSION_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_GET_SOFTWARE_VERSION_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_GET_SOFTWARE_VERSION_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
/**
 * @brief Get firmware version of the reader/writer module
 * @param ppVersion_out Address where the output version string pointer will be stored
 */
uhfman_err_t uhfman_get_software_version(uhfman_ctx_t* pCtx, char** ppcVersion_out);

#define UHFMAN_GET_MANUFACTURER_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_GET_MANUFACTURER_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_GET_MANUFACTURER_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_GET_MANUFACTURER_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_GET_MANUFACTURER_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_GET_MANUFACTURER_VERSION_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_GET_MANUFACTURER_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
/**
 * @brief Get manufacturer of the reader/writer module
 * @param ppManufacturer_out Address where the output manufacturer string pointer will be stored
 */
uhfman_err_t uhfman_get_manufacturer(uhfman_ctx_t* pCtx, char** ppcManufacturer_out);

#define UHFMAN_GET_SELECT_PARAM_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_GET_SELECT_PARAM_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_GET_SELECT_PARAM_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_GET_SELECT_PARAM_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_GET_SELECT_PARAM_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_GET_SELECT_PARAM_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_GET_SELECT_PARAM_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
/**
 * @brief Get Select parameter
 */
uhfman_err_t uhfman_dbg_get_select_param(uhfman_ctx_t* pCtx);

#define UHFMAN_GET_QUERY_PARAMS_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_GET_QUERY_PARAMS_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_GET_QUERY_PARAMS_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_GET_QUERY_PARAMS_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_GET_QUERY_PARAMS_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_GET_QUERY_PARAMS_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_GET_QUERY_PARAMS_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
/**
 * @brief Get Query Parameters
 */
uhfman_err_t uhfman_dbg_get_query_params(uhfman_ctx_t* pCtx);

#define UHFMAN_GET_WORKING_CHANNEL_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_GET_WORKING_CHANNEL_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_GET_WORKING_CHANNEL_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_GET_WORKING_CHANNEL_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_GET_WORKING_CHANNEL_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_GET_WORKING_CHANNEL_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_GET_WORKING_CHANNEL_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
/**
 * @brief Get working channel
 */
uhfman_err_t uhfman_dbg_get_working_channel(uhfman_ctx_t* pCtx);

#define UHFMAN_GET_WORK_AREA_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_GET_WORK_AREA_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_GET_WORK_AREA_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_GET_WORK_AREA_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_GET_WORK_AREA_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_GET_WORK_AREA_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_GET_WORK_AREA_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
/**
 * @brief Get work area
 */
uhfman_err_t uhfman_dbg_get_work_area(uhfman_ctx_t* pCtx);

#define UHFMAN_GET_TRANSMIT_POWER_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_GET_TRANSMIT_POWER_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_GET_TRANSMIT_POWER_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_GET_TRANSMIT_POWER_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_GET_TRANSMIT_POWER_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_GET_TRANSMIT_POWER_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_GET_TRANSMIT_POWER_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
/**
 * @brief Get transmit power
 */
uhfman_err_t uhfman_dbg_get_transmit_power(uhfman_ctx_t* pCtx);

#define UHFMAN_GET_DEMOD_PARAMS_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_GET_DEMOD_PARAMS_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_GET_DEMOD_PARAMS_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_GET_DEMOD_PARAMS_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_GET_DEMOD_PARAMS_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_GET_DEMOD_PARAMS_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_GET_DEMOD_PARAMS_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
/**
 * @brief Get receiver demodulator parameters
 */
uhfman_err_t uhfman_dbg_get_demod_params(uhfman_ctx_t* pCtx);

#define UHFMAN_SINGLE_POLLING_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_SINGLE_POLLING_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_SINGLE_POLLING_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_SINGLE_POLLING_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_SINGLE_POLLING_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_SINGLE_POLLING_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_SINGLE_POLLING_ERR_READ_NOTIFICATION UHFMAN_ERR_READ_NOTIFICATION
#define UHFMAN_SINGLE_POLLING_ERR_UNEXPECTED_FRAME_TYPE UHFMAN_ERR_UNEXPECTED_FRAME_TYPE
#define UHFMAN_SINGLE_POLLING_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
uhfman_err_t uhfman_dbg_single_polling(uhfman_ctx_t* pCtx);

#define UHFMAN_MULTIPLE_POLLING_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_MULTIPLE_POLLING_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_MULTIPLE_POLLING_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_MULTIPLE_POLLING_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_MULTIPLE_POLLING_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_MULTIPLE_POLLING_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_MULTIPLE_POLLING_ERR_READ_NOTIFICATION UHFMAN_ERR_READ_NOTIFICATION
#define UHFMAN_MULTIPLE_POLLING_ERR_UNEXPECTED_FRAME_TYPE UHFMAN_ERR_UNEXPECTED_FRAME_TYPE
#define UHFMAN_MULTIPLE_POLLING_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
uhfman_err_t uhfman_dbg_multiple_polling(uhfman_ctx_t* pCtx);

//TODO add & implement more functions