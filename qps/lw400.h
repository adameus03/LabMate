/**
 *  @file lw400.h
 * 
 *  @brief Low-level functions for sending commands to Dymo LabelWriter 400
 *  Based on the technical reference provided by Dymo. The function descriptions are extracts from the documentation provided by Dymo 
 *  @see https://download.dymo.com/UserManuals/LabelWriter%20400%20Series%20Tech%20Ref_4_08.pdf
 * 
 */

#include "printer_common.h"

#define LW400_ESC_B_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ESC_B_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * @brief "Set Dot Tab". This command shifts the starting dot position on the print head towards the right,
 * effectively introducing an increased left margin. Each byte represents 8 dots, so a value
 * of four for n would shift an image over 32 dots, or 32/300ths of an inch.
 * @param n Starting byte number per line (binary), where 0 <= n <= 83, default value = 0. 
 */
printer_err_t lw400_esc_B(printer_ctx_t* pCtx, uint8_t n);

#define LW400_ESC_D_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ESC_D_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * @brief "Set Bytes per Line". This command reduces the number of bytes sent for each line if the right side of the label
 * is to be left blank. 
 * @param n Number of bytes per line, where 1 <= n <= 84, default value = 84
 */
printer_err_t lw400_esc_D(printer_ctx_t* pCtx, uint8_t n);

#define LW400_ESC_L_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ESC_L_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * @brief "Set Label Length". This command indicates the maximum distance the printer should travel while searching
 * for the top-of-form hole or mark. Print lines and lines fed both count towards this total so
 * that a value related to the length of the label to be printed can be used.
 * For normal labels with top-of-form marks, the actual distance fed is adjusted once the
 * top–of-form mark is detected. As a result this command is usually set to a value slightly
 * longer than the true label length to ensure that the top-of-form mark is reached before
 * feeding is terminated.
 * This command can also be used to put the printer into continuous feed mode. Any
 * negative value (0x8000 - 0xFFFF) will place the printer in continuous feed mode.
 * In continuous feed mode the printer will respond to Form Feed (<esc> E) and Short Form
 * Feed (<esc> G) commands by feeding a few lines out from the current print position. An
 * ESC E command causes the print position to feed to the tear bar and an ESC G causes it
 * to feed far enough so that a reverse feed will not cause lines to overlap. 
 * @param labelLength Number of dot lines from sense hole to sense hole (binary), default value = 3058 (10.2")
 */
printer_err_t lw400_esc_L(printer_ctx_t* pCtx, int16_t labelLength);

#define LW400_ESC_E_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ESC_E_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * @brief "Form Feed". This command advances the most recently printed label to a position where it can be torn
 * off. This positioning places the next label beyond the starting print position. Therefore, a
 * reverse-feed will be automatically invoked when printing on the next label. To optimize
 * print speed and eliminate this reverse feeding when printing multiple labels, use the Short
 * Form Feed command between labels, and the Form Feed command after the
 * last label.
 */
printer_err_t lw400_esc_E(printer_ctx_t* pCtx);

#define LW400_ESC_G_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ESC_G_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * @brief "Short Form Feed". This command feeds the next label into print position. The most recently printed label
will still be partially inside the printer and cannot be torn off. This command is meant to
be used only between labels on a multiple label print job. 
 */
printer_err_t lw400_esc_G(printer_ctx_t* pCtx);

#define LW400_ESC_A_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ESC_A_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
#define LW400_ESC_A_ERR_READ_RESPONSE PRINTER_ERR_READ_RESPONSE
/**
 * @brief "Get Printer Status"
 * @param pStatus_out pointer to output value: single byte with the following bit meanings (1 = true):
 * Bit 0 (lsb) - Ready (paper in, no jam)
 * Bit 1       - Top of form
 * Bit 2       - Not used
 * Bit 3       - Not used
 * Bit 4       - Not used
 * Bit 5       - No paper
 * Bit 6       - Paper jam
 * Bit 7       - Printer error (jam, invalid sequence, and so on)
 *  
 * @note Printer ready is returned as 03h (Ready and Top of form). 
 */
printer_err_t lw400_esc_A(printer_ctx_t* pCtx, uint8_t* pStatus_out);

#define LW400_ESC_at_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ESC_at_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * @brief "Reset Printer". This command resets all parameters (Dot Tabs, Line Tabs, Bytes per Line, and so on) to
 * their default values and sets top-of-form as true.
   @note This command is acted upon immediately; any data still in the print buffer will be
 * lost.
 */
printer_err_t lw400_esc_at(printer_ctx_t* pCtx);

#define LW400_ESC_star_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ESC_star_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * @brief "Restore Default Settings". This command resets all internal parameters to their default values. 
 * @note This command is acted upon when it is received
 */
printer_err_t lw400_esc_star(printer_ctx_t* pCtx);

#define LW400_ESC_f_x01_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ESC_f_x01_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * @brief This command skips over the next “n” lines on the label.
 * @param n the number of lines to skip 
 * @note This command is unusual because it requires 0x01 prior to the value for the
number of lines to skip.
 */
printer_err_t lw400_esc_f_x01(printer_ctx_t* pCtx, uint8_t n);

#define LW400_ESC_V_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ESC_V_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
#define LW400_ESC_V_ERR_READ_RESPONSE PRINTER_ERR_READ_RESPONSE
/**
 * @brief "Return Revision Letter/Number". This command returns the printer model and firmware version number as an ASCII
 * string. The information is returned as an 8-character ASCII string in the following
 * format: Bytes 0-4: the 5-digit model number (e.g. "93089"), Byte 5: a lowercase letter (commonly "v"), Bytes 6-7: the two-digit firmware version (e.g. "0N"), Example: 98039v0K
 * @param pRevision_out pointer to output value (You should allocate 8 bytes for this)
*/
printer_err_t lw400_esc_V(printer_ctx_t* pCtx, uint8_t* pRevision_out);

#define LW400_SYN_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_SYN_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * @brief "Transfer Print Data". This command sends a raster line of print data to the printer.
 * @param pData Print data (bit "1" represens a blackened dot)
 */
printer_err_t lw400_syn(printer_ctx_t* pCtx, uint8_t* pData, uint8_t len);

#define LW400_ETB_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ETB_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * "Transfer Compressed Print Data". Won't be implemented/used for this project (at least for now)
 */
printer_err_t lw400_etb(printer_ctx_t* pCtx, uint8_t* data, uint8_t len);

#define LW400_ESC_h_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ESC_h_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * @brief "Text Speed Mode" This command instructs the printer to print in Text Quality mode. This is the default, high
 * speed printing mode. This command instructs the printer to print in Barcode and Graphics mode. This results in
 * lower speed but greater positional and sizing accuracy of the print elements. 
 */
printer_err_t lw400_esc_h(printer_ctx_t* pCtx);

#define LW400_ESC_i_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ESC_i_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * @brief "Barcode and Graphics Mode". 
 */
printer_err_t lw400_esc_i(printer_ctx_t* pCtx);

#define LW400_ESC_c_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ESC_c_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * @brief "Set Print Density Light". This command sets the strobe time of the printer to 75% of its standard duty cycle.
 */
printer_err_t lw400_esc_c(printer_ctx_t* pCtx);

#define LW400_ESC_d_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ESC_d_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * @brief "Set Print Density Medium". This command sets the strobe time of the printer to 87.5% of its standard duty cycle.
 */
printer_err_t lw400_esc_d(printer_ctx_t* pCtx);

#define LW400_ESC_e_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ESC_e_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * @brief "Set Print Density Normal". This command sets the strobe time of the printer to 100% of its standard duty cycle.
 */
printer_err_t lw400_esc_e(printer_ctx_t* pCtx);

#define LW400_ESC_g_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ESC_g_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * @brief "Set Print Density Dark". This command sets the strobe time of the printer to 112.5% of its standard duty cycle.
 */
printer_err_t lw400_esc_g(printer_ctx_t* pCtx);

#define LW400_ESC_y_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ESC_y_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * @brief "Set print resolution to 300x300". This command sets the step resolution of the printer to match its print element resolution,
giving 300 x 300 dpi printing.
 */
printer_err_t lw400_esc_y(printer_ctx_t* pCtx);

#define LW400_ESC_z_ERR_SUCCESS PRINTER_ERR_SUCCESS
#define LW400_ESC_z_ERR_SEND_COMMAND PRINTER_ERR_SEND_COMMAND
/**
 * @brief "Set print resolution to 203x300". This command changes the step resolution of the printer to 203 dpi giving a printing
resolution of 203 x 300 dpi. 
 */
printer_err_t lw400_esc_z(printer_ctx_t* pCtx);

