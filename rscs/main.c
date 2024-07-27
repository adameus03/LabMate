#include <stdio.h>
#include <stdlib.h>
#include "plibsys.h"
#include <psocket.h>
#include <plibsysconfig.h>
#include <pmacros.h>
#include <ptypes.h>
#include "uhfman.h"

int main() {
    fprintf(stdout, "-------- RSCS --------\n");
    uhfman_ctx_t uhfmanCtx = {};
    fprintf(stdout, "Calling uhfman_device_take\n");
    uhfman_err_t err = uhfman_device_take(&uhfmanCtx);
    if (err != UHFMAN_TAKE_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "uhfman_device_take returned %d\n", err);
        return 1;
    }
    fprintf(stdout, "uhfman_device_take returned successfully\n");

    fprintf(stdout, "Calling uhfman_get_hardware_version\n");
    char* hardwareVersion = NULL;
    err = uhfman_get_hardware_version(&uhfmanCtx, &hardwareVersion);
    if (err != UHFMAN_GET_HARDWARE_VERSION_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_get_hardware_version returned %d\n", err);
        //return 1;
    }
    if (hardwareVersion == NULL) {
        fprintf(stderr, "Hardware version is NULL\n");
    } else {
        fprintf(stdout, "Hardware version: %s\n", hardwareVersion);
    }

    fprintf(stdout, "Calling uhfman_get_software_version\n");
    char* softwareVersion = NULL;
    err = uhfman_get_software_version(&uhfmanCtx, &softwareVersion);
    if (err != UHFMAN_GET_SOFTWARE_VERSION_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_get_software_version returned %d\n", err);
        //return 1;
    }
    if (softwareVersion == NULL) {
        fprintf(stderr, "Software version is NULL\n");
    } else {
        fprintf(stdout, "Software version: %s\n", softwareVersion);
    }

    fprintf(stdout, "Calling uhfman_device_release\n");
    uhfman_device_release(&uhfmanCtx);
    fprintf(stdout, "uhfman_device_release returned\n");

    return 0;
}


// #include <errno.h>
// #include <signal.h>
// #include <string.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <stdint.h>
// #include <unistd.h>
// #include <sys/select.h>
// #include <termios.h>

// #include <libusb.h>

// #define EP_DATA_IN        (0x2|LIBUSB_ENDPOINT_IN)
// #define EP_DATA_OUT       (0x2|LIBUSB_ENDPOINT_OUT)
// #define CTRL_IN           (LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_IN)
// #define CTRL_OUT          (LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_ENDPOINT_OUT)
// #define DEFAULT_BAUD_RATE 9600

// static struct libusb_device_handle *devh = NULL;
// static struct libusb_transfer *recv_bulk_transfer = NULL;
// uint8_t dtr = 0;
// uint8_t rts = 0;
// uint8_t do_exit = 0;
// uint8_t recvbuf[1024];

// void writeHandshakeByte(void) {
//     if (libusb_control_transfer(devh, CTRL_OUT, 0xa4, ~((dtr ? 1 << 5 : 0) | (rts ? 1 << 6 : 0)), 0, NULL, 0, 1000) < 0) {
//         fprintf(stderr, "Faild to set handshake byte\n");
//     }
// }

// int setBaudRate(int baudRate){
//     static int baud[] = {2400, 0xd901, 0x0038, 4800, 0x6402,
//             0x001f, 9600, 0xb202, 0x0013, 19200, 0xd902, 0x000d, 38400,
//             0x6403, 0x000a, 115200, 0xcc03, 0x0008};

//     for (int i = 0; i < sizeof(baud)/sizeof(int) / 3; i++) {
//         if (baud[i * 3] == baudRate) {
//             int r = libusb_control_transfer(devh, CTRL_OUT, 0x9a, 0x1312, baud[i * 3 + 1], NULL, 0, 1000);
//             if (r < 0) {
//                 fprintf(stderr, "failed control transfer 0x9a,0x1312\n");
//                 return r;
//             }
//             r = libusb_control_transfer(devh, CTRL_OUT, 0x9a, 0x0f2c, baud[i * 3 + 2], NULL, 0, 1000);
//             if (r < 0) {
//                 fprintf(stderr, "failed control transfer 0x9a,0x0f2c\n");
//                 return r;
//             }

//             return 0;
//         }
//     }
//     fprintf(stderr, "unsupported baudrate\n");
//     return -1;
// }

// int init_ch34x()
// {
//     int r;

//     r = libusb_control_transfer(devh, CTRL_OUT, 0xa1, 0, 0, NULL, 0, 1000);
//     if (r < 0) {
//         fprintf(stderr, "failed control transfer 0xa1\n");
//         return r;
//     }
//     r = libusb_control_transfer(devh, CTRL_OUT, 0x9a, 0x2518, 0x0050, NULL, 0, 1000);
//     if (r < 0) {
//         fprintf(stderr, "failed control transfer 0x9a,0x2518\n");
//         return r;
//     }
//     r = libusb_control_transfer(devh, CTRL_OUT, 0xa1, 0x501f, 0xd90a, NULL, 0, 1000);
//     if (r < 0) {
//         fprintf(stderr, "failed control transfer 0xa1,0x501f\n");
//         return r;
//     }

//     setBaudRate(DEFAULT_BAUD_RATE);
//     writeHandshakeByte();

//     return r;
// }

// static void LIBUSB_CALL cb_img(struct libusb_transfer *transfer)
// {
//     if (transfer->status != LIBUSB_TRANSFER_COMPLETED) {
//         fprintf(stderr, "img transfer status %d?\n", transfer->status);
//         do_exit = 2;
//         libusb_free_transfer(transfer);
//         return;
//     }

//     // printf("Data callback[");
//     for (int i = 0; i < transfer->actual_length; ++i)
//     {
//         putchar(recvbuf[i]);
//     }
//     fflush(stdout);
//     // printf("]\n");

//     if (libusb_submit_transfer(recv_bulk_transfer) < 0)
//         do_exit = 2;
// }

// int send_to_uart(void)
// {
//     printf("send_to_uart\n");
//     int r;
//     unsigned char sendbuf[1024];
//     if ((r = read(0, sendbuf, sizeof(sendbuf))) < 0) {
//         return r;
//     } else {
//         int transferred, len = r;
//         r = libusb_bulk_transfer(devh, EP_DATA_OUT, sendbuf, len, &transferred, 200);
//         // printf("read[%d]transferred[%d]\n", len, transferred);
//         if(r < 0){
//             fprintf(stderr, "libusb_bulk_transfer error %d\n", r);
//             return r;
//         }
//     }
//     return r;
// }

// int kbhit()
// {
//     printf("in kbhit\n");
//     struct timeval tv = { 0L, 0L };
//     fd_set fds;
//     FD_ZERO(&fds);
//     FD_SET(0, &fds);
//     return select(1, &fds, NULL, NULL, &tv);
// }

// int main(int argc, char **argv)
// {
//     int r = 1;

//     r = libusb_init(NULL);
//     if (r < 0) {
//         fprintf(stderr, "failed to initialise libusb\n");
//         exit(1);
//     }

//     devh = libusb_open_device_with_vid_pid(NULL, 0x1a86, 0x7523);
//     if (devh == NULL) {
//         fprintf(stderr, "Could not find/open device\n");
//         goto out;
//     }

//     r = libusb_detach_kernel_driver(devh, 0);
//     if (r < 0) {
//         fprintf(stderr, "libusb_detach_kernel_driver error %d\n", r);
//         goto out;
//     }

//     r = libusb_claim_interface(devh, 0);
//     if (r < 0) {
//         fprintf(stderr, "libusb_claim_interface error %d\n", r);
//         goto out;
//     }
//     printf("claimed interface\n");

//     r = init_ch34x();
//     if (r < 0)
//         goto out_release;

//     if(argc > 1)
//         setBaudRate(atoi(argv[1]));

//     printf("initialized\n");

//     recv_bulk_transfer = libusb_alloc_transfer(0);
//     if (!recv_bulk_transfer){
//         fprintf(stderr, "libusb_alloc_transfer error\n");
//         goto out_release;
//     }
//     printf("allocated transfer\n");

//     libusb_fill_bulk_transfer(recv_bulk_transfer, devh, EP_DATA_IN, recvbuf,
//         sizeof(recvbuf), cb_img, NULL, 0);
//     printf("filled transfer\n");

//     r = libusb_submit_transfer(recv_bulk_transfer);
//     if (r < 0){
//         fprintf(stderr, "libusb_submit_transfer error\n");
//         goto out_deinit;
//     }
//     printf("submitted transfer\n");

//     // set_conio_terminal_mode();

//     printf("Looping to handle events\n");
//     while (!do_exit) {
//         printf("in while\n");
//         struct timeval tv = { 0L, 0L };
//         r = libusb_handle_events_timeout(NULL, &tv);
//         printf("libusb_handle_events_timeout returned %d\n", r);
//         if (r < 0) {
//             printf("Going to out_deinit\n");
//             goto out_deinit;
//         }
//         if(kbhit()){
//             printf("kbhit\n");
//             r = send_to_uart();
//             if (r < 0)
//                 goto out_deinit;
//         }
//     }

//     if (recv_bulk_transfer) {
//         r = libusb_cancel_transfer(recv_bulk_transfer);
//         if (r < 0)
//             goto out_deinit;
//     }

// out_deinit:
//     libusb_free_transfer(recv_bulk_transfer);
// out_release:
//     libusb_release_interface(devh, 0);
// out:
//     libusb_close(devh);
//     libusb_exit(NULL);
//     return r >= 0 ? r : -r;
// }