#include <stdio.h>
#include <stdlib.h>
#include "plibsys.h"
#include <psocket.h>
#include <plibsysconfig.h>
#include <pmacros.h>
#include <ptypes.h>

#include "printer.h"
#include "qr_data_adapter.h"

// TODO NOW: Adjust to changed printer.h API
// TODO: Abstract away plibsys calls? 

#define IP_ADDR "127.0.0.1"
#define PORT 3535

#define RID_SIZE 64

#define QPS_STATUS_USB_FAIL -4
#define QPS_STATUS_QR_GENERATION_FAILED -3
#define QPS_STATUS_BUSY -2
#define QPS_STATUS_INVALID_RID_LENGTH -1
#define QPS_STATUS_READY 0
#define QPS_STATUS_FINISHED 1

static PMutex* pMutex = NULL;

/**
 * @return TRUE if an error was handled, FALSE otherwise.
 */
static pboolean qps_handle_error(PError *error, pboolean shouldExit) {
    if (error != NULL) {
        P_ERROR(p_error_get_message(error));
        p_error_free(error);
        if (shouldExit) { exit(EXIT_FAILURE); }
        return TRUE;
    }
    return FALSE;
}

/**
 * @return TRUE if an error was handled, FALSE otherwise.
 */
static pboolean qps_handle_error_boolean(PError *error, pboolean result, pboolean shouldExit) {
    if (!result) {
        qps_handle_error(error, shouldExit);
        if (!shouldExit) { return TRUE; }
        P_ERROR("Error occurred, but no error object was provided.");
        if (shouldExit) { exit(EXIT_FAILURE); }
        return TRUE;
    }
    return FALSE;
}

static const char* qps_get_status_symbol(int status) {
    switch (status) {
        case QPS_STATUS_USB_FAIL:
            return "QPS_STATUS_USB_FAIL";
        case QPS_STATUS_QR_GENERATION_FAILED:
            return "QPS_STATUS_QR_GENERATION_FAILED";
        case QPS_STATUS_BUSY:
            return "PQPS_STATUS_BUSY";
        case QPS_STATUS_INVALID_RID_LENGTH:
            return "QPS_STATUS_INVALID_RID_LENGTH";
        case QPS_STATUS_READY:
            return "QPS_STATUS_READY";
        case QPS_STATUS_FINISHED:
            return "QPS_STATUS_FINISHED";
        default:
            return "Unknown status";
    }
}

/**
 * @returns TRUE if the status was sent to the client successfully, FALSE otherwise.
 */
static pboolean qps_server_send_status(PSocket* pClientSocket, int status) {
    PError* error = NULL;
    int rv = p_socket_send(pClientSocket, (const pchar*)&status, sizeof(int), &error);
    if (qps_handle_error_boolean(error, rv != -1, FALSE)) {
        const char* statusSymbol = qps_get_status_symbol(status);
        fprintf(stderr, "[Failed to send status %s]", statusSymbol);
        return FALSE;
    }
    return TRUE;
}

static void qps_nop() {}
#define BREAK_IF_FALSE(rv) if (!(rv)) { break; } qps_nop() 

static void qps_stall(int n, int ms) {
    for (int i = 0; i < n; i++) {
        printf("Stalling for %d ms (%d / %d)\n", ms, i + 1, n);
        p_uthread_sleep(ms);
    }
    
}

static void qps_handle_client(PSocket* pClientSocket) {
    PError* error = NULL;
    PSocketAddress* pClientSocketAddress = p_socket_get_remote_address(pClientSocket, &error);
    if (qps_handle_error(error, FALSE)) {
        P_ERROR("[Failed to get client address]");
        p_socket_free(pClientSocket);
        return;
    }

    pchar* ip = p_socket_address_get_address(pClientSocketAddress);
    puint16 port = p_socket_address_get_port(pClientSocketAddress);

    printf("Client connected from %s:%d\n", ip, port);

    while (TRUE) {
        // Receive RID (64 bytes)
        pchar* rid = p_malloc(RID_SIZE);
        if (rid == NULL) {
            P_ERROR("[Failed to allocate memory for RID]");
            break;
        }
        for (int i = 0; i < RID_SIZE; i++) {
            rid[i] = 0;
        }
        
        pssize rv = p_socket_receive(pClientSocket, rid, RID_SIZE, &error);
        if (qps_handle_error_boolean(error, rv != -1, FALSE)) {
            P_ERROR("[Failed to receive RID]");
            break;
        }
        if (rv != RID_SIZE) {
            if (!qps_server_send_status(pClientSocket, QPS_STATUS_INVALID_RID_LENGTH)) {
                break;
            }
        } else {
            if(p_mutex_trylock(pMutex)) {
                BREAK_IF_FALSE(qps_server_send_status(pClientSocket, QPS_STATUS_READY));

                // qps_stall(3, 1000);

                P_DEBUG("Calling printer_take");
                printer_ctx_t ctx = {};
                int rv = printer_take(&ctx);
                if (rv != PRINTER_TAKE_ERR_SUCCESS) {
                    P_ERROR("USB related error");
                    fprintf(stdout, "printer_take returned %d\n", rv);
                    BREAK_IF_FALSE(qps_server_send_status(pClientSocket, QPS_STATUS_USB_FAIL));
                    break; // Failed to connect to the printer so it doesn't make sense to proceed; the client has been notified hopefully
                }

                // P_DEBUG("Calling printer_esc_V");
                // rv = printer_esc_V(&ctx);
                // fprintf(stdout, "printer_esc_V returned %d\n", rv);

                P_DEBUG("Calling printer_get_revision");
                char revision[PRINTER_REVISION_STRING_LENGTH + 1];
                rv = printer_get_revision(&ctx, revision);
                if (rv != PRINTER_GET_REVISION_ERR_SUCCESS) {
                    P_ERROR("Failed to get printer revision");
                    fprintf(stdout, "printer_get_revision returned %d\n", rv);
                    BREAK_IF_FALSE(qps_server_send_status(pClientSocket, QPS_STATUS_USB_FAIL));
                    break;
                }
                fprintf(stdout, "Printer revision: %s\n", revision);

                P_DEBUG("Calling printer_setup");
                rv = printer_setup(&ctx);
                if (rv != PRINTER_SETUP_ERR_SUCCESS) {
                    P_ERROR("Failed to setup printer");
                    fprintf(stdout, "printer_setup returned %d\n", rv);
                    BREAK_IF_FALSE(qps_server_send_status(pClientSocket, QPS_STATUS_USB_FAIL));
                    break;
                }

                P_DEBUG("Converting to QR code");
                uint8_t* pGrayscaleData = NULL;
                int nGrayscaleDataWidth = 0;
                int nGrayscaleDataHeight = 0;
                rv = qda_qrencu8buf_to_grayscale((const uint8_t*)rid, RID_SIZE, &pGrayscaleData, &nGrayscaleDataWidth, &nGrayscaleDataHeight);
                if (rv != QDA_QRENCU8BUF2BMP_ERR_SUCCESS) {
                    P_ERROR("Failed to convert to QR code");
                    fprintf(stderr, "qda_dlw500u8buf_to_grayscale returned %d\n", rv);
                    p_free(rid);
                    BREAK_IF_FALSE(qps_server_send_status(pClientSocket, QPS_STATUS_QR_GENERATION_FAILED));
                }
                P_DEBUG("Finished converting to QR code");
                p_free(rid);

                #if QR_DATA_ADAPTER_USE_DEBUG_EXTENSIONS == 1
                fprintf(stdout, "--- DEBUG EXTENSIONS ARE ENABLED ---\n");
                qda_grayscale_print_to_console(pGrayscaleData, nGrayscaleDataWidth, nGrayscaleDataHeight);
                uint8_t* pGrayscaleExpandedPixelsData = NULL;
                int nGrayscaleExpandedPixelsDataWidth = 0;
                int nGrayscaleExpandedPixelsDataHeight = 0;
                qda_grayscale_expand_pixels(pGrayscaleData, nGrayscaleDataWidth, nGrayscaleDataHeight, &pGrayscaleExpandedPixelsData, &nGrayscaleExpandedPixelsDataWidth, &nGrayscaleExpandedPixelsDataHeight, 4);
                //p_free(pGrayscaleData);

                uint8_t* pGrayscaleExpandedPaddedData = NULL;
                int nGrayscaleExpandedPaddedDataWidth = 0;
                int nGrayscaleExpandedPaddedDataHeight = 0;
                qda_grayscale_pad(QDA_GRAYSCALE_PAD_MODE_ALL_SIDES, pGrayscaleExpandedPixelsData, nGrayscaleExpandedPixelsDataWidth, nGrayscaleExpandedPixelsDataHeight, &pGrayscaleExpandedPaddedData, &nGrayscaleExpandedPaddedDataWidth, &nGrayscaleExpandedPaddedDataHeight, 4);
                p_free(pGrayscaleExpandedPixelsData);

                uint8_t* pRgbData = NULL;
                int nRgbDataLen = 0;
                qda_grayscale_to_rgb(pGrayscaleExpandedPaddedData, nGrayscaleExpandedPaddedDataWidth, nGrayscaleExpandedPaddedDataHeight, &pRgbData, &nRgbDataLen);
                p_free(pGrayscaleExpandedPaddedData);

                qda_rgb_save_to_bmp_file(pRgbData, nGrayscaleExpandedPaddedDataWidth, nGrayscaleExpandedPaddedDataHeight, 3, "qr.bmp");
                #endif


                P_DEBUG("CALLING printer_print");
                // P_DEBUG("Calling printer_esc_d");
                // rv = printer_esc_d(&ctx, pGrayscaleData, nGrayscaleDataWidth, nGrayscaleDataHeight);
                // fprintf(stdout, "printer_esc_d returned %d\n", rv);
                rv = printer_print(&ctx, pGrayscaleData, nGrayscaleDataWidth * nGrayscaleDataHeight);
                if (rv != PRINTER_PRINT_ERR_SUCCESS) {
                    P_ERROR("Failed to print");
                    fprintf(stdout, "printer_print returned %d\n", rv);
                    BREAK_IF_FALSE(qps_server_send_status(pClientSocket, QPS_STATUS_USB_FAIL)); // TODO: Change/add error codes (not only this one, but also the other few - look lines above)
                    break;
                }
                P_DEBUG("Done printing");
                P_DEBUG("Finished work");
                p_free(pGrayscaleData);

                printer_release(&ctx);

                BREAK_IF_FALSE(qps_server_send_status(pClientSocket, QPS_STATUS_FINISHED));

                if (!p_mutex_unlock(pMutex)) {
                    P_ERROR("Failed to unlock mutex");
                    exit(EXIT_FAILURE);
                }
            } else {
                BREAK_IF_FALSE(qps_server_send_status(pClientSocket, QPS_STATUS_BUSY));
            }
        }
    }


    p_free(ip);
    p_socket_address_free(pClientSocketAddress);
    p_socket_free(pClientSocket);
}

int main() {
    p_libsys_init();

    pMutex = p_mutex_new();

    PError *error = NULL;
    pboolean result = FALSE;
    PSocket* pSocket = p_socket_new(P_SOCKET_FAMILY_INET, P_SOCKET_TYPE_STREAM, P_SOCKET_PROTOCOL_TCP, &error);
    qps_handle_error(error, TRUE);

    PSocketAddress* pSocketAddress = p_socket_address_new(IP_ADDR, PORT);
    
    result = p_socket_bind(pSocket, pSocketAddress, FALSE, &error);
    qps_handle_error_boolean(error, result, TRUE);

    result = p_socket_listen(pSocket, &error);
    qps_handle_error_boolean(error, result, TRUE);

    while (TRUE) {
        PSocket* pClientSocket = p_socket_accept(pSocket, &error);
        if (qps_handle_error(error, FALSE)) {
            P_ERROR("Failed to accept client socket");
            continue;
        }

        PUThread *pThread = p_uthread_create((PUThreadFunc)qps_handle_client, pClientSocket, FALSE, NULL);
        if (pThread == NULL) {
            P_ERROR("Failed to create thread for client");
            p_socket_free(pClientSocket);
            continue;
        }
    }
}
