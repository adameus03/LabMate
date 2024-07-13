#include <stdio.h>
#include <stdlib.h>
#include "plibsys.h"
#include <psocket.h>
#include <plibsysconfig.h>
#include <pmacros.h>
#include <ptypes.h>

#include "printer.h"

#define IP_ADDR "127.0.0.1"
#define PORT 3535

#define RID_SIZE 64

#define QPS_STATUS_BUSY -2
#define QPS_STATUS_INVALID_RID_LENGTH -1
#define QPS_STATUS_READY 0
#define QPS_STATUS_FINISHED 1

static PMutex* pMutex = NULL;

/**
 * @return TRUE if an error was handled, FALSE otherwise.
 */
static pboolean handle_error(PError *error, pboolean shouldExit) {
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
static pboolean handle_error_boolean(PError *error, pboolean result, pboolean shouldExit) {
    if (!result) {
        handle_error(error, shouldExit);
        if (!shouldExit) { return TRUE; }
        P_ERROR("Error occurred, but no error object was provided.");
        if (shouldExit) { exit(EXIT_FAILURE); }
        return TRUE;
    }
    return FALSE;
}

static void handle_client(PSocket* pClientSocket) {
    PError* error = NULL;
    PSocketAddress* pClientSocketAddress = p_socket_get_remote_address(pClientSocket, &error);
    if (handle_error(error, FALSE)) {
        P_ERROR("[Failed to get client address]");
        p_socket_free(pClientSocket);
        return;
    }

    pchar* ip = p_socket_address_get_address(pClientSocketAddress);
    puint16 port = p_socket_address_get_port(pClientSocketAddress);

    printf("Client connected from %s:%d\n", ip, port);

    while (TRUE) {
        // Receive RID (64 bytes)
        static pchar rid[RID_SIZE];
        pssize rv = p_socket_receive(pClientSocket, rid, RID_SIZE, &error);
        if (handle_error_boolean(error, rv != -1, FALSE)) {
            P_ERROR("[Failed to receive RID]");
            break;
        }
        if (rv != RID_SIZE) {
            int status = QPS_STATUS_INVALID_RID_LENGTH;
            rv = p_socket_send(pClientSocket, (const pchar*)&status, sizeof(int), &error);
            if (handle_error_boolean(error, rv != -1, FALSE)) {
                P_ERROR("[Failed to send  status QPS_STATUS_INVALID_RID_LENGTH]");
                break;
            }
        } else {
            if(p_mutex_trylock(pMutex)) {
                int status = QPS_STATUS_READY;
                rv = p_socket_send(pClientSocket, (const pchar*)&status, sizeof(int), &error);
                if (handle_error_boolean(error, rv != -1, FALSE)) {
                    P_ERROR("[Failed to send status QPS_STATUS_READY]");
                    break;
                }

                // Do some work - sleeps for now
                P_DEBUG("Doing some work 1");
                p_uthread_sleep(1000);
                P_DEBUG("Doing some work 2");
                p_uthread_sleep(1000);
                P_DEBUG("Doing some work 3");
                p_uthread_sleep(1000);
                P_DEBUG("Calling print_test");
                int rv = print_test();
                fprintf(stdout, "print_test returned %d\n", rv);
                P_DEBUG("Finished work");


                status = QPS_STATUS_FINISHED;
                rv = p_socket_send(pClientSocket, (const pchar*)&status, sizeof(int), &error);
                if (handle_error_boolean(error, rv != -1, FALSE)) {
                    P_ERROR("[Failed to send status QPS_STATUS_FINISHED]");
                    break;
                }

                if (!p_mutex_unlock(pMutex)) {
                    P_ERROR("Failed to unlock mutex");
                    exit(EXIT_FAILURE);
                }
            } else {
                int status = QPS_STATUS_BUSY;
                rv = p_socket_send(pClientSocket, (const pchar*)&status, sizeof(int), &error);
                if (handle_error_boolean(error, rv != -1, FALSE)) {
                    P_ERROR("[Failed to send status QPS_STATUS_BUSY]");
                    break;
                }
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
    handle_error(error, TRUE);

    PSocketAddress* pSocketAddress = p_socket_address_new(IP_ADDR, PORT);
    
    result = p_socket_bind(pSocket, pSocketAddress, FALSE, &error);
    handle_error_boolean(error, result, TRUE);

    result = p_socket_listen(pSocket, &error);
    handle_error_boolean(error, result, TRUE);

    while (TRUE) {
        PSocket* pClientSocket = p_socket_accept(pSocket, &error);
        if (handle_error(error, FALSE)) {
            P_ERROR("Failed to accept client socket");
            continue;
        }

        PUThread *pThread = p_uthread_create((PUThreadFunc)handle_client, pClientSocket, FALSE, NULL);
        if (pThread == NULL) {
            P_ERROR("Failed to create thread for client");
            p_socket_free(pClientSocket);
            continue;
        }
    }
}
