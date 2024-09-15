#include <stdio.h>
#include <stdlib.h>
#include "plibsys.h"
#include <psocket.h>
#include <plibsysconfig.h>
#include <pmacros.h>
#include <ptypes.h>
#include <assert.h>
#include "uhfman.h"

void main_uhfman_poll_handler(uint16_t handle) {
    // uhfman_tag_t tag = uhfman_tag_get(handle);
    // fprintf(stdout, "Tag %u, EPC: ", tag.handle);
    // for (uint32_t j = 0; j < YPDR200_X22_NTF_PARAM_EPC_LENGTH; j++) {
    //     fprintf(stdout, "%02X ", tag.epc[j]);
    // }
    // fprintf(stdout, "\n");
    // return;


    uhfman_tag_t* pTags = NULL;
    uint32_t nTags = 0;
    uhfman_list_tags(&pTags, &nTags);
    uhfman_tag_stats_t* pStats = (uhfman_tag_stats_t*)malloc(nTags * sizeof(uhfman_tag_stats_t));

    for (uint32_t i = 0; i < nTags; i++) {
        uhfman_tag_t tag = pTags[i];
        // fprintf(stdout, "Tag %d: ", i);
        // for (uint32_t j = 0; j < YPDR200_X22_NTF_PARAM_EPC_LENGTH; j++) {
        //     fprintf(stdout, "%02X", tag.epc[j]);
        // }
        // fprintf(stdout, "\n");
        pStats[i] = uhfman_tag_get_stats(tag.handle);
    }

    for (uint32_t i = 0; i < nTags; i++) {
        // fprintf(stdout, "Handle %u: , ", pTags[i].handle);
        // fprintf(stdout, "EPC: ");
        // for (uint32_t j = 0; j < YPDR200_X22_NTF_PARAM_EPC_LENGTH; j++) {
        //     fprintf(stdout, "%02X", pTags[i].epc[j]);
        // }
        // fprintf(stdout, ", ");
        fprintf(stdout, "%4.2f %lu   ", pStats[i].rssi_avg_per_period, pStats[i].read_time_interval_avg_per_period);
    }
    fprintf(stdout, "\n");
}

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

    fprintf(stdout, "Calling uhfman_get_manufacturer\n");
    char* manufacturer = NULL;
    err = uhfman_get_manufacturer(&uhfmanCtx, &manufacturer);
    if (err != UHFMAN_GET_MANUFACTURER_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_get_manufacturer returned %d\n", err);
        //return 1;
    } else {
        fprintf(stdout, "Manufacturer: %s\n", manufacturer);
    }

    fprintf(stdout, "Calling uhfman_set_select_param\n");
    uint8_t target = UHFMAN_SELECT_TARGET_SL;
    uint8_t action = uhfman_select_action(UHFMAN_SEL_SL_ASSERT, UHFMAN_SEL_SL_DEASSERT);
    assert(action != UHFMAN_SELECT_ACTION_INVALID);
    printf("Select action = 0x%02X\n", action);
    uint8_t memBank = UHFMAN_SELECT_MEMBANK_EPC;
    uint32_t ptr = 0x20;
    uint8_t maskLen = 0x60;
    uint8_t truncate = UHFMAN_SELECT_TRUNCATION_DISABLED;
    const uint8_t mask[12] = {
        //0xE2, 0x80, 0x69, 0x15, 0x00, 0x00, 0x40, 0x17, 0xAA, 0xE6, 0x69, 0xBC
        //0xE2, 0x80, 0x69, 0x15, 0x00, 0x00, 0x40, 0x17, 0xAA, 0xE6, 0x69, 0xBD
        //0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA 
        //0xE2, 0x80, 0x68, 0x94, 0x00, 0x00, 0x40, 0x24, 0xED, 0x64, 0x21, 0x84 
        0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB
    };

    err = uhfman_set_select_param(&uhfmanCtx, target, action, memBank, ptr, maskLen, truncate, mask);
    if (err != UHFMAN_SET_SELECT_PARAM_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_set_select_param returned %d\n", err);
        //return 1;
    } else {
        fprintf(stdout, "uhfman_set_select_param returned successfully\n");
    }

    fprintf(stdout, "Calling uhfman_get_select_param\n");
    err = uhfman_dbg_get_select_param(&uhfmanCtx);
    if (err != UHFMAN_GET_SELECT_PARAM_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_get_select_param returned %d\n", err);
        //return 1;
    } else {
        fprintf(stdout, "uhfman_get_select_param returned successfully\n");
    }
    //exit(0);

    fprintf(stdout, "!!! Calling uhfman_set_query_params !!!\n");
    err = uhfman_set_query_params(&uhfmanCtx, UHFMAN_QUERY_SEL_SL, UHFMAN_QUERY_SESSION_S0, UHFMAN_QUERY_TARGET_A, 0x00);
    if (err != UHFMAN_SET_QUERY_PARAMS_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_set_query_params returned %d\n", err);
        //return 1;
    } else {
        fprintf(stdout, "uhfman_set_query_params returned successfully\n");
    }

    fprintf(stdout, "Calling uhfman_dbg_get_query_params\n");
    err = uhfman_dbg_get_query_params(&uhfmanCtx);
    if (err != UHFMAN_GET_QUERY_PARAMS_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_dbg_get_query_params returned %d\n", err);
        //return 1;
    } else {
        fprintf(stdout, "uhfman_dbg_get_query_params returned successfully\n");
    }

    fprintf(stdout, "Calling uhfman_dbg_get_working_channel\n");
    err = uhfman_dbg_get_working_channel(&uhfmanCtx);
    if (err != UHFMAN_GET_WORKING_CHANNEL_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_dbg_get_working_channel returned %d\n", err);
        //return 1;
    } else {
        fprintf(stdout, "uhfman_dbg_get_working_channel returned successfully\n");
    }

    fprintf(stdout, "Calling uhfman_get_work_area\n");
    err = uhfman_dbg_get_work_area(&uhfmanCtx);
    if (err != UHFMAN_GET_WORK_AREA_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_get_work_area returned %d\n", err);
        //return 1;
    } else {
        fprintf(stdout, "uhfman_get_work_area returned successfully\n");
    }


    // fprintf(stdout, "!!! Calling uhfman_set_transmit_power !!!\n");
    // err = uhfman_set_transmit_power(&uhfmanCtx, 15.0f);
    // if (err != UHFMAN_SET_TRANSMIT_POWER_ERR_SUCCESS) {
    //     P_ERROR("USB related error");
    //     fprintf(stderr, "ERROR (ignoring): uhfman_set_transmit_power returned %d\n", err);
    //     //return 1;
    // } else {
    //     fprintf(stdout, "uhfman_set_transmit_power returned successfully\n");
    // }


    fprintf(stdout, "Calling uhfman_dbg_get_transmit_power\n");
    err = uhfman_dbg_get_transmit_power(&uhfmanCtx);
    if (err != UHFMAN_GET_TRANSMIT_POWER_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_dbg_get_transmit_power returned %d\n", err);
        //return 1;
    } else {
        fprintf(stdout, "uhfman_dbg_get_transmit_power returned successfully\n");
    }

    fprintf(stdout, "Calling uhfman_get_demod_params\n");
    err = uhfman_dbg_get_demod_params(&uhfmanCtx);
    if (err != UHFMAN_GET_DEMOD_PARAMS_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_get_demod_params returned %d\n", err);
        //return 1;
    } else {
        fprintf(stdout, "uhfman_get_demod_params returned successfully\n");
    }

    fprintf(stdout, "Calling uhfman_set_select_mode\n");
    err = uhfman_set_select_mode(&uhfmanCtx, UHFMAN_SELECT_MODE_ALWAYS);
    if (err != UHFMAN_SET_SELECT_MODE_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_set_select_mode returned %d\n", err);
        //return 1;
    } else {
        fprintf(stdout, "uhfman_set_select_mode returned successfully\n");
    }

    // fprintf(stdout, "Calling uhfman_write_tag_mem\n");
    // const uint8_t access_password[4] = {
    //     0x00, 0x00, 0x00, 0x00
    // };
    // uhfman_tag_mem_bank_t mem_bank = UHFMAN_TAG_MEM_BANK_EPC;
    // uint16_t wordPtr = UHFMAN_TAG_MEM_EPC_WORD_PTR_EPC;
    // uint16_t wordCount = UHFMAN_TAG_MEM_EPC_WORD_COUNT_EPC;
    // const uint8_t epc[12] = {
    //     //0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA 
    //     //0xE2, 0x80, 0x69, 0x15, 0x00, 0x00, 0x40, 0x17, 0xAA, 0xE6, 0x69, 0xBD
    //     //0xE2, 0x80, 0x69, 0x15, 0x00, 0x00, 0x40, 0x17, 0xAA, 0xE6, 0x69, 0xBC
    //     0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB
    // };
    // uint16_t pc = 0xFFFF;
    // uint8_t* pEPC = NULL;
    // size_t epc_len = 0;
    // uint8_t resp_err = 0;
    // err = uhfman_write_tag_mem(&uhfmanCtx, access_password, mem_bank, wordPtr, wordCount, epc, &pc, &pEPC, &epc_len, &resp_err);
    // if (err != UHFMAN_WRITE_TAG_MEM_ERR_SUCCESS) {
    //     if (err == UHFMAN_WRITE_TAG_MEM_ERR_ERROR_RESPONSE) {
    //         if (resp_err == UHFMAN_TAG_ERR_ACCESS_DENIED) {
    //             //P_ERROR("Error response obtained from tag");
    //             P_ERROR("Access denied error when trying to write tag's memory (most probably the provided access password was invalid)");
    //             fprintf(stdout, "pc = 0x%04X, epc_len = %lu\n", pc, epc_len);
    //             fprintf(stdout, "EPC: ");
    //             for (size_t i = 0; i < epc_len; i++) {
    //                 fprintf(stdout, "%02X ", pEPC[i]);
    //             }
    //             fprintf(stdout, "\n");
    //             free (pEPC);
    //         }
    //     } else {
    //         P_ERROR("USB related error"); // TODO improve those error messages, theses are not always really neccessarily USB related, but rather related to underlying UHF RFID interrogator module
    //     }
    //     fprintf(stderr, "ERROR (ignoring): uhfman_write_tag_mem returned %d\n", err);
    //     //return 1;
    // } else {
    //     fprintf(stdout, "pc = 0x%04X, epc_len = %lu\n", pc, epc_len);
    //     fprintf(stdout, "EPC: ");
    //     for (size_t i = 0; i < epc_len; i++) {
    //         fprintf(stdout, "%02X ", pEPC[i]);
    //     }
    //     fprintf(stdout, "\n");
    //     free (pEPC);
    //     fprintf(stdout, "uhfman_write_tag_mem returned successfully\n");
    // }
    // exit(0);

    // fprintf(stdout, "Calling uhfman_dbg_single_polling\n");
    // err = uhfman_dbg_single_polling(&uhfmanCtx);
    // if (err != UHFMAN_SINGLE_POLLING_ERR_SUCCESS) {
    //     P_ERROR("USB related error"); // TODO improve those error messages, theses are not really neccessarily USB related, but rather related to underlying UHF RFID interrogator module
    //     fprintf(stderr, "ERROR (ignoring): uhfman_dbg_single_polling returned %d\n", err);
    //     //return 1;
    // } else {
    //     fprintf(stdout, "uhfman_dbg_single_polling returned successfully\n");
    // }

    fprintf(stdout, "Calling uhfman_set_poll_handler\n");
    ufhman_set_poll_handler(main_uhfman_poll_handler);

    fprintf(stdout, "Calling uhfman_dbg_multiple_polling\n");
    err = uhfman_dbg_multiple_polling(&uhfmanCtx);
    if (err != UHFMAN_MULTIPLE_POLLING_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_dbg_multiple_polling returned %d\n", err);
        //return 1;
    } else {
        fprintf(stdout, "uhfman_dbg_multiple_polling returned successfully\n");
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