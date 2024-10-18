/**
 * @file uhfman.h
 * @brief UHF RFID reader/writer management
 */

/*
    bash/client software --- database
         |
    (filesystem)
         |
       main (FUSE)
         |
       uhfd
        |
      uhfman <<< we are here
        |
     ypdr200
*/

#ifndef UHFMAN_H
#define UHFMAN_H

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

#define UHFMAN_SELECT_TARGET_S0 0x00
#define UHFMAN_SELECT_TARGET_S1 0x01
#define UHFMAN_SELECT_TARGET_S2 0x02
#define UHFMAN_SELECT_TARGET_S3 0x03
#define UHFMAN_SELECT_TARGET_SL 0x04
#define UHFMAN_SELECT_TARGET_RFU0 0x05
#define UHFMAN_SELECT_TARGET_RFU1 0x06
#define UHFMAN_SELECT_TARGET_RFU2 0x07
#define UHFMAN_SELECT_TARGET_UNKNOWN 0xFF

#define UHFMAN_SEL_SL_ASSERT 0
#define UHFMAN_SEL_SL_DEASSERT 1
#define UHFMAN_SEL_SL_NEGATE 2
#define UHFMAN_SEL_NOP 3
#define UHFMAN_SEL_INVEN_A 0
#define UHFMAN_SEL_INVEN_B 1
#define UHFMAN_SEL_INVEN_TOGGLE 2

#define UHFMAN_SELECT_ACTION_UNKNOWN 0xFF

uint8_t uhfman_select_action(uint8_t uTagMatching, uint8_t uTagNotMatching);

#define UHFMAN_SELECT_MEMBANK_FILETYPE 0x00
#define UHFMAN_SELECT_MEMBANK_EPC 0x01
#define UHFMAN_SELECT_MEMBANK_TID 0x02
#define UHFMAN_SELECT_MEMBANK_FILE_0 0x03
#define UHFMAN_SELECT_MEMBANK_UNKNOWN 0xFF

#define UHFMAN_SELECT_TRUNCATION_DISABLED 0x00
#define UHFMAN_SELECT_TRUNCATION_ENABLED 0x01

#define UHFMAN_SET_SELECT_PARAM_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_SET_SELECT_PARAM_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_SET_SELECT_PARAM_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_SET_SELECT_PARAM_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_SET_SELECT_PARAM_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_SET_SELECT_PARAM_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_SET_SELECT_PARAM_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
/**
 * @brief Set Select parameter
 */
uhfman_err_t uhfman_set_select_param(uhfman_ctx_t* pCtx, 
                                     uint8_t target, 
                                     uint8_t action, 
                                     uint8_t memBank, 
                                     uint32_t ptr, 
                                     uint8_t maskLen, 
                                     uint8_t truncate, 
                                     const uint8_t* pMask);

// #define UHFMAN_SET_SELECT_PARAM_BY_EPC_ERR_SUCCESS UHFMAN_ERR_SUCCESS
// #define UHFMAN_SET_SELECT_PARAM_BY_EPC_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
// #define UHFMAN_SET_SELECT_PARAM_BY_EPC_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
// #define UHFMAN_SET_SELECT_PARAM_BY_EPC_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
// #define UHFMAN_SET_SELECT_PARAM_BY_EPC_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN
// #define UHFMAN_SET_SELECT_PARAM_BY_EPC_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
// #define UHFMAN_SET_SELECT_PARAM_BY_EPC_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
// 
// @brief Set Select parameter (using EPC)
// 
// uhfman_err_t uhfman_set_select_param_by_epc_code(uhfman_ctx_t* pCtx, const uint8_t* pEPC, size_t epcLen);


#define UHFMAN_SET_SELECT_MODE_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_SET_SELECT_MODE_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_SET_SELECT_MODE_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_SET_SELECT_MODE_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_SET_SELECT_MODE_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_SET_SELECT_MODE_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_SET_SELECT_MODE_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
/**
 * @brief Set Select mode
 * @param mode Determines when the Select command will be issued by the interrogator
 */
uhfman_err_t uhfman_set_select_mode(uhfman_ctx_t* pCtx, uhfman_select_mode_t mode);

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


#define UHFMAN_SET_QUERY_PARAMS_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_SET_QUERY_PARAMS_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_SET_QUERY_PARAMS_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_SET_QUERY_PARAMS_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_SET_QUERY_PARAMS_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_SET_QUERY_PARAMS_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_SET_QUERY_PARAMS_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
/**
 * @brief Set Query Parameters
 * @param sel Selects which tags are to respond to the Query command
 * @param session Chooses the session for the inventory round
 * @param target selects whether Tags whose inventoried flag is A or B participate in the inventory round
 * @param q sets the number of slots in the round for the slot counter algorithm
 */
uhfman_err_t uhfman_set_query_params(uhfman_ctx_t* pCtx, uhfman_query_sel_t sel, uhfman_query_session_t session, uhfman_query_target_t target, uint8_t q);

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

#define UHFMAN_SET_TRANSMIT_POWER_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_SET_TRANSMIT_POWER_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_SET_TRANSMIT_POWER_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_SET_TRANSMIT_POWER_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_SET_TRANSMIT_POWER_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_SET_TRANSMIT_POWER_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_SET_TRANSMIT_POWER_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
/**
 * @brief Set transmit power
 * @param txPower Transmit power in dBm
 */
uhfman_err_t uhfman_set_transmit_power(uhfman_ctx_t* pCtx, float txPower);

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

//typedef void (*uhfman_inventory_callback_t)(void* pUserData, const uint8_t* pEPC, size_t epcLen);

#define UHFMAN_SINGLE_POLLING_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_SINGLE_POLLING_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_SINGLE_POLLING_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_SINGLE_POLLING_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_SINGLE_POLLING_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_SINGLE_POLLING_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_SINGLE_POLLING_ERR_READ_NOTIFICATION UHFMAN_ERR_READ_NOTIFICATION
#define UHFMAN_SINGLE_POLLING_ERR_UNEXPECTED_FRAME_TYPE UHFMAN_ERR_UNEXPECTED_FRAME_TYPE
#define UHFMAN_SINGLE_POLLING_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
uhfman_err_t uhfman_single_polling(uhfman_ctx_t* pCtx, void* pUserData);

#define UHFMAN_MULTIPLE_POLLING_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_MULTIPLE_POLLING_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_MULTIPLE_POLLING_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_MULTIPLE_POLLING_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_MULTIPLE_POLLING_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_MULTIPLE_POLLING_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_MULTIPLE_POLLING_ERR_READ_NOTIFICATION UHFMAN_ERR_READ_NOTIFICATION
#define UHFMAN_MULTIPLE_POLLING_ERR_UNEXPECTED_FRAME_TYPE UHFMAN_ERR_UNEXPECTED_FRAME_TYPE
#define UHFMAN_MULTIPLE_POLLING_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
uhfman_err_t uhfman_multiple_polling(uhfman_ctx_t* pCtx, unsigned long timeout_us, void* pUserData);

#define UHFMAN_MULTIPLE_POLLING_STOP_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_MULTIPLE_POLLING_STOP_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_MULTIPLE_POLLING_STOP_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_MULTIPLE_POLLING_STOP_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_MULTIPLE_POLLING_STOP_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_MULTIPLE_POLLING_STOP_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_MULTIPLE_POLLING_STOP_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
uhfman_err_t uhfman_multiple_polling_stop(uhfman_ctx_t* pCtx);

typedef enum {
    UHFMAN_TAG_MEM_BANK_RESERVED = 0x00,
    UHFMAN_TAG_MEM_BANK_EPC = 0x01,
    UHFMAN_TAG_MEM_BANK_TID = 0x02,
    UHFMAN_TAG_MEM_BANK_USER = 0x03
} uhfman_tag_mem_bank_t;

#define UHFMAN_TAG_MEM_RESERVED_WORD_PTR_KILL_PASSWD 0x0
#define UHFMAN_TAG_MEM_RESERVED_WORD_PTR_ACCESS_PASSWD 0x2
#define UHFMAN_TAG_MEM_EPC_WORD_PTR_STORED_PC 0x1
#define UHFMAN_TAG_MEM_EPC_WORD_PTR_EPC 0x2

#define UHFMAN_TAG_MEM_RESERVED_WORD_COUNT_KILL_PASSWD 0x2
#define UHFMAN_TAG_MEM_RESERVED_WORD_COUNT_ACCESS_PASSWD 0x2
#define UHFMAN_TAG_MEM_EPC_WORD_COUNT_STORED_PC 0x1
#define UHFMAN_TAG_MEM_EPC_WORD_COUNT_EPC 0x6

// #define UHFMAN_TAG_ERR_SUCCESS 0x00
// #define UHFMAN_TAG_ERR_ACCESS_DENIED 0x16
// #define UHFMAN_TAG_ERR_UNKNOWN 0xFF

#define UHFMAN_WRITE_TAG_MEM_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_WRITE_TAG_MEM_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_WRITE_TAG_MEM_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_WRITE_TAG_MEM_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_WRITE_TAG_MEM_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_WRITE_TAG_MEM_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_WRITE_TAG_MEM_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
/**
 * @brief Write to currently interrogated tag's memory
 * @param accessPasswd Access password for performing IO on the tag memory
 * @param memBank Memory bank to write to
 * @param wordPtr Starting word target address
 * @param nWords Number of words to write
 * @param pData Pointer to the data to write
 * @param pPC_out Pointer to the PC value of the tag which was written to, can be NULL
 * @param ppEPC_out Pointer to the address where to store the EPC of the tag which was written to, can be NULL. If not NULL, the caller is responsible for freeing the memory allocated for the EPC by this function
 * @param pEPC_len_out Pointer to the length of the EPC of the tag which was written to, can be NULL
 * @param pRespErrCode_out Address where to store error code read from error response frame, can be NULL
 */
uhfman_err_t uhfman_write_tag_mem(uhfman_ctx_t* pCtx, 
                                  const uint8_t accessPasswd[4], 
                                  uhfman_tag_mem_bank_t memBank, 
                                  uint16_t wordPtr, 
                                  uint16_t nWords, 
                                  const uint8_t* pData,
                                  uint16_t* pPC_out,
                                  uint8_t** ppEPC_out,
                                  size_t* pEPC_len_out,
                                  uint8_t* pRespErrCode_out);

#define UHFMAN_LOCK_TAG_MEM_FILE_0_PERMALOCK        (1U<<0)
#define UHFMAN_LOCK_TAG_MEM_FILE_0_PWD_WRITE            (1U<<1)
#define UHFMAN_LOCK_TAG_MEM_TID_PERMALOCK           (1U<<2)
#define UHFMAN_LOCK_TAG_MEM_TID_PWD_WRITE               (1U<<3)
#define UHFMAN_LOCK_TAG_MEM_EPC_PERMALOCK           (1U<<4)
#define UHFMAN_LOCK_TAG_MEM_EPC_PWD_WRITE               (1U<<5)
#define UHFMAN_LOCK_TAG_MEM_ACCESS_PASSWD_PERMALOCK (1U<<6)
#define UHFMAN_LOCK_TAG_MEM_ACCESS_PASSWD_PWD_RW        (1U<<7)
#define UHFMAN_LOCK_TAG_MEM_KILL_PASSWD_PERMALOCK   (1U<<8)
#define UHFMAN_LOCK_TAG_MEM_KILL_PASSWD_PWD_RW          (1U<<9)

#define UHFMAN_LOCK_TAG_MEM_FILE_0_MASK_PERMALOCK UHFMAN_LOCK_TAG_MEM_FILE_0_PERMALOCK
#define UHFMAN_LOCK_TAG_MEM_FILE_0_MASK_PWD_WRITE UHFMAN_LOCK_TAG_MEM_FILE_0_PWD_WRITE
#define UHFMAN_LOCK_TAG_MEM_TID_MASK_PERMALOCK UHFMAN_LOCK_TAG_MEM_TID_PERMALOCK
#define UHFMAN_LOCK_TAG_MEM_TID_MASK_PWD_WRITE UHFMAN_LOCK_TAG_MEM_TID_PWD_WRITE
#define UHFMAN_LOCK_TAG_MEM_EPC_MASK_PERMALOCK UHFMAN_LOCK_TAG_MEM_EPC_PERMALOCK
#define UHFMAN_LOCK_TAG_MEM_EPC_MASK_PWD_WRITE UHFMAN_LOCK_TAG_MEM_EPC_PWD_WRITE
#define UHFMAN_LOCK_TAG_MEM_ACCESS_PASSWD_MASK_PERMALOCK UHFMAN_LOCK_TAG_MEM_ACCESS_PASSWD_PERMALOCK
#define UHFMAN_LOCK_TAG_MEM_ACCESS_PASSWD_MASK_PWD_RW UHFMAN_LOCK_TAG_MEM_ACCESS_PASSWD_PWD_RW
#define UHFMAN_LOCK_TAG_MEM_KILL_PASSWD_MASK_PERMALOCK UHFMAN_LOCK_TAG_MEM_KILL_PASSWD_PERMALOCK
#define UHFMAN_LOCK_TAG_MEM_KILL_PASSWD_MASK_PWD_RW UHFMAN_LOCK_TAG_MEM_KILL_PASSWD_PWD_RW

#define UHFMAN_LOCK_TAG_MEM_FILE_0_ACTION_PERMALOCK UHFMAN_LOCK_TAG_MEM_FILE_0_PERMALOCK
#define UHFMAN_LOCK_TAG_MEM_FILE_0_ACTION_PWD_WRITE UHFMAN_LOCK_TAG_MEM_FILE_0_PWD_WRITE
#define UHFMAN_LOCK_TAG_MEM_TID_ACTION_PERMALOCK UHFMAN_LOCK_TAG_MEM_TID_PERMALOCK
#define UHFMAN_LOCK_TAG_MEM_TID_ACTION_PWD_WRITE UHFMAN_LOCK_TAG_MEM_TID_PWD_WRITE
#define UHFMAN_LOCK_TAG_MEM_EPC_ACTION_PERMALOCK UHFMAN_LOCK_TAG_MEM_EPC_PERMALOCK
#define UHFMAN_LOCK_TAG_MEM_EPC_ACTION_PWD_WRITE UHFMAN_LOCK_TAG_MEM_EPC_PWD_WRITE
#define UHFMAN_LOCK_TAG_MEM_ACCESS_PASSWD_ACTION_PERMALOCK UHFMAN_LOCK_TAG_MEM_ACCESS_PASSWD_PERMALOCK
#define UHFMAN_LOCK_TAG_MEM_ACCESS_PASSWD_ACTION_PWD_RW UHFMAN_LOCK_TAG_MEM_ACCESS_PASSWD_PWD_RW
#define UHFMAN_LOCK_TAG_MEM_KILL_PASSWD_ACTION_PERMALOCK UHFMAN_LOCK_TAG_MEM_KILL_PASSWD_PERMALOCK
#define UHFMAN_LOCK_TAG_MEM_KILL_PASSWD_ACTION_PWD_RW UHFMAN_LOCK_TAG_MEM_KILL_PASSWD_PWD_RW

#define UHFMAN_LOCK_TAG_MEM_ERR_SUCCESS UHFMAN_ERR_SUCCESS
#define UHFMAN_LOCK_TAG_MEM_ERR_SEND_COMMAND UHFMAN_ERR_SEND_COMMAND
#define UHFMAN_LOCK_TAG_MEM_ERR_READ_RESPONSE UHFMAN_ERR_READ_RESPONSE
#define UHFMAN_LOCK_TAG_MEM_ERR_NOT_SUPPORTED UHFMAN_ERR_NOT_SUPPORTED
#define UHFMAN_LOCK_TAG_MEM_ERR_UNKNOWN_DEVICE_MODEL UHFMAN_ERR_UNKNOWN_DEVICE_MODEL
#define UHFMAN_LOCK_TAG_MEM_ERR_ERROR_RESPONSE UHFMAN_ERR_ERROR_RESPONSE
#define UHFMAN_LOCK_TAG_MEM_ERR_UNKNOWN UHFMAN_ERR_UNKNOWN
/**
 * @brief Lock tag memory
 * @param accessPasswd Access password to perform the lock operation
 * @param lock_mask_flags Lock mask flags (see UHFMAN_LOCK_TAG_MEM_*_MASK_*)
 * @param lock_action_flags Lock action flags (see UHFMAN_LOCK_TAG_MEM_*_ACTION_*)
 * @param pPC_out Pointer to the PC value of the tag which was locked, can be NULL
 * @param ppEPC_out Pointer to the address where to store the EPC of the tag which was locked, can be NULL. If not NULL, the caller is responsible for freeing the memory allocated for the EPC by this function
 * @param pEPC_len_out Pointer to the length of the EPC of the tag which was locked, can be NULL
 * @param pRespErrCode_out Address where to store error code read from error response frame, can be NULL
 */
uhfman_err_t uhfman_lock_tag_mem(uhfman_ctx_t* pCtx, 
                                 const uint8_t accessPasswd[4],
                                 uint16_t lock_mask_flags, 
                                 uint16_t lock_action_flags,
                                 uint16_t* pPC_out,
                                 uint8_t** ppEPC_out,
                                 size_t* pEPC_len_out,
                                 uint8_t* pRespErrCode_out);

//TODO add & implement more functions

//----------------------------------------------

//TODO adjust UHFMAN_TAG_PERIOD_NREADS
// TODO make dynamic in the future (replace with dynamic mem allocation) --> more robust especially when using UHFMAN_POLL_MODE_RAW
/* Affects the stats collecting when using UHFMAN_POLL_MODE_TRACK. When using UHFMAN_POLL_MODE_RAW it should be set big enough to handle most frequent reads in multiple polling. */
#define UHFMAN_TAG_PERIOD_NREADS 100/*40*/ /*5*/ /*10*/ /*2*/
// TODO Make UHFMAN_MAX_NUM_TAGS dynamic in the future (replace with dynamic mem allocation)
#define UHFMAN_MAX_NUM_TAGS 500 
typedef struct {
    uint16_t handle;
    uint8_t epc[YPDR200_X22_NTF_PARAM_EPC_LENGTH];
    uint32_t num_reads;
    //uint32_t read_times[UHFMAN_TAG_PERIOD_NREADS];
    unsigned long read_times[UHFMAN_TAG_PERIOD_NREADS];
    uint8_t rssi[UHFMAN_TAG_PERIOD_NREADS];
} uhfman_tag_t;

typedef struct {
    float rssi_avg_per_period;
    uint32_t read_time_interval_avg_per_period;
} uhfman_tag_stats_t;

void uhfman_list_tags(uhfman_tag_t** ppTags, uint32_t* pnTags_out);

typedef void (*uhfman_tag_handler_t)(uhfman_tag_t tag, void* pUserData);
typedef void (*uhfman_poll_handler_t)(uint16_t handle, void* pUserData);
void uhfman_set_new_tag_event_handler(uhfman_tag_handler_t handler);

void uhfman_unset_new_tag_event_handler();

void uhfman_set_poll_handler(uhfman_poll_handler_t handler);

void uhfman_unset_poll_handler();

void uhfman_tag_anonymous_forget(); // TODO what about passing the context parameter here and in similar functions? Should we pass it and why?

typedef enum {
  /* Raw mode: the tag is not tracked, only the EPC is read and rssi is measured. Computationally leightweight. */
  UHFMAN_POLL_MODE_RAW = 0,
  /* Track mode: the tag is tracked, the EPC is read, rssi is measured and the tag is stored in an internal array which increases in size as new tags are discovered. Searching the array when updating statistics increases computational cost. (TODO unless tags were given sequential EPC or devno numbers were stored in tag memory) */
  UHFMAN_POLL_MODE_TRACK = 1
} uhfman_poll_mode_t;

void uhfman_set_poll_mode(const uhfman_poll_mode_t mode);

typedef enum {
  UHFMAN_TIME_PRECISION_MS = 0,
  UHFMAN_TIME_PRECISION_US = 1,
  UHFMAN_TIME_PRECISION_NS = 2
} uhfman_time_precision_t;

void uhfman_set_time_precision(const uhfman_time_precision_t precision);

uhfman_tag_t uhfman_tag_get(uint16_t handle); // TODO threading, locking

uhfman_tag_stats_t uhfman_tag_get_stats(uint16_t handle);

#endif // UHFMAN_H
