#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>
// TODO: Fix this
//#if PRINTER_USE_DEBUG_EXTENSIONS == 1
//#include <string.h> // For strerror
//#endif
#include <string.h> // For strerror

#include "lw400.h"

#define DYMO_ENDPONT_ADDR_IN 0x82U
#define DYMO_ENDPONT_ADDR_OUT 0x02U

#define BULK_TRANSFER_TIMEOUT_MS 5000

//#define DYMO_VENDOR_ID 0x046d
//#define DYMO_PRODUCT_ID 0xc534


// DONE: Separate libusb and device initialization into separate functions - most functions should be called assuming the device is already initialized
// TODO: Abstract away libusb_bulk_transfer calls into a function that takes a direction, data, and length, so the code is more library-agnostic
// TODO: Replace NULLs in libusb_bulk_transfer to handle unexpected errors? (not sure if this is necessary at all)
// TODO: Remove repeating result handling code and debug printfs (optionally replace those with logging)


printer_err_t lw400_esc_B(printer_ctx_t* pCtx, uint8_t n) {
    libusb_device_handle* pHandle = pCtx->handle;
    assert(n <= 83U);
    unsigned char data_out[] = {0x1BU, 0x42U, n};

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    switch (r) {
        case 0:
            printf("ESC_B command sent successfully\n");
            return LW400_ESC_B_ERR_SUCCESS;
        default:
            fprintf(stderr, "Error sending ESC_B command: %s\n", libusb_error_name(r));
            return LW400_ESC_B_ERR_SEND_COMMAND;
    }
}

printer_err_t lw400_esc_D(printer_ctx_t* pCtx, uint8_t n) {
    libusb_device_handle* pHandle = pCtx->handle;
    assert(n >= 1U && n <= 84U);
    unsigned char data_out[] = {0x1BU, 0x44U, n};

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    switch (r) {
        case 0:
            printf("ESC_D command sent successfully\n");
            return LW400_ESC_D_ERR_SUCCESS;
        default:
            fprintf(stderr, "ESC_D error sending command: %s\n", libusb_error_name(r));
            return LW400_ESC_D_ERR_SEND_COMMAND;
    }

}

printer_err_t lw400_esc_L(printer_ctx_t* pCtx, int16_t labelLength) {
    libusb_device_handle* pHandle = pCtx->handle;
    uint8_t data_out[] = {
        0x1BU, 
        0x4CU, 
        (uint8_t)(labelLength >> 8), // MSB
        (uint8_t)(labelLength & 0xFFU) // LSB
    };

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    switch (r) {
        case 0:
            printf("ESC_L command sent successfully\n");
            return LW400_ESC_L_ERR_SUCCESS;
        default:
            fprintf(stderr, "Error sending ESC_L command: %s\n", libusb_error_name(r));
            return LW400_ESC_L_ERR_SEND_COMMAND;
    }

}

printer_err_t lw400_esc_E(printer_ctx_t* pCtx) {
    libusb_device_handle* pHandle = pCtx->handle;
    unsigned char data_out[] = {0x1BU, 0x45U};

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    switch (r) {
        case 0:
            printf("ESC_E command sent successfully\n");
            return LW400_ESC_E_ERR_SUCCESS;
        default:
            fprintf(stderr, "Error sending ESC_E command: %s\n", libusb_error_name(r));
            return LW400_ESC_E_ERR_SEND_COMMAND;
    }

}

printer_err_t lw400_esc_G(printer_ctx_t* pCtx) {
    libusb_device_handle* pHandle = pCtx->handle;
    unsigned char data_out[] = {0x1BU, 0x47U};

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    switch (r) {
        case 0:
            printf("ESC_G command sent successfully\n");
            return LW400_ESC_G_ERR_SUCCESS;
        default:
            fprintf(stderr, "Error sending ESC_G command: %s\n", libusb_error_name(r));
            return LW400_ESC_G_ERR_SEND_COMMAND;
    }

}

printer_err_t lw400_esc_A(printer_ctx_t* pCtx, uint8_t* pStatus_out) {
    libusb_device_handle* pHandle = pCtx->handle;
    unsigned char data_out[] = {0x1BU, 0x41U};
    unsigned char data_in[1];
    int actual_length;

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    if (r != 0) {
        fprintf(stderr, "Error sending ESC_A  command: %s\n", libusb_error_name(r));
        return LW400_ESC_A_ERR_SEND_COMMAND;
    }
    printf("ESC_A command sent successfully\n");

    r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_IN, data_in, sizeof(data_in), &actual_length, BULK_TRANSFER_TIMEOUT_MS);
    if (r != 0) {
        fprintf(stderr, "Error reading response for ESC_A command: %s\n", libusb_error_name(r));
        return LW400_ESC_A_ERR_READ_RESPONSE;
    }
    printf("ESC_A response received: %02X\n", data_in[0]);

    *pStatus_out = data_in[0];
    return LW400_ESC_A_ERR_SUCCESS;

}

printer_err_t lw400_esc_at(printer_ctx_t* pCtx) {
    libusb_device_handle* pHandle = pCtx->handle;
    unsigned char data_out[] = {0x1BU, 0x40U};

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    switch (r) {
        case 0:
            printf("ESC_@ command sent successfully\n");
            return LW400_ESC_at_ERR_SUCCESS;
        default:
            fprintf(stderr, "Error sending ESC_@ command: %s\n", libusb_error_name(r));
            return LW400_ESC_at_ERR_SEND_COMMAND;
    }

}

printer_err_t lw400_esc_star(printer_ctx_t* pCtx) {
    libusb_device_handle* pHandle = pCtx->handle;
    unsigned char data_out[] = {0x1BU, 0x2AU};

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    switch (r) {
        case 0:
            printf("ESC_* command sent successfully\n");
            return LW400_ESC_star_ERR_SUCCESS;
        default:
            fprintf(stderr, "Error sending ESC_* command: %s\n", libusb_error_name(r));
            return LW400_ESC_star_ERR_SEND_COMMAND;
    }

}

printer_err_t lw400_esc_f_x01(printer_ctx_t* pCtx, uint8_t n) {
    libusb_device_handle* pHandle = pCtx->handle;
    unsigned char data_out[] = {0x1BU, 0x66U, 0x01U, n};

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    switch (r) {
        case 0:
            printf("ESC_f_x01 command sent successfully\n");
            return LW400_ESC_f_x01_ERR_SUCCESS;
        default:
            fprintf(stderr, "Error sending ESC_f_x01 command: %s\n", libusb_error_name(r));
            return LW400_ESC_f_x01_ERR_SEND_COMMAND;
    }


}

printer_err_t lw400_esc_V(printer_ctx_t* pCtx, uint8_t* pRevision_out) {
    libusb_device_handle* pHandle = pCtx->handle;

    unsigned char data_out[] = {0x1B, 0x56};
    unsigned char data_in[8];
    int actual_length;

    // Send the command to the printer
    //int r = libusb_bulk_transfer(handle, (1 | LIBUSB_ENDPOINT_OUT), data_out, sizeof(data_out), &actual_length, 0);
    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), &actual_length, 0);
    if (r == 0 && actual_length == sizeof(data_out)) {
        printf("ESC_V command sent successfully\n");
    } else {
        fprintf(stderr, "Error sending ESC_V command: %s\n", libusb_error_name(r));
        return LW400_ESC_V_ERR_SEND_COMMAND;
    }

    // Read the response from the printer
    //r = libusb_bulk_transfer(handle, (1 | LIBUSB_ENDPOINT_IN), data_in, sizeof(data_in), &actual_length, 5000); // 5 second timeout
    r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_IN, data_in, sizeof(data_in), &actual_length, BULK_TRANSFER_TIMEOUT_MS); // 5 second timeout
    if (r == 0) {
        printf("Response received for command ESC_V:\n");
        for (int i = 0; i < actual_length; i++) {
            printf("%02X ", data_in[i]);
        }
        printf("\n");
    } else {
        fprintf(stderr, "Error reading response for command ESC_V: %s\n", libusb_error_name(r));
        return LW400_ESC_V_ERR_READ_RESPONSE;
    }

    // Copy the response to the output buffer
    memcpy(pRevision_out, data_in, sizeof(data_in));
    return LW400_ESC_V_ERR_SUCCESS;
}

printer_err_t lw400_syn(printer_ctx_t* pCtx, uint8_t* pData, uint8_t len) {
    libusb_device_handle* pHandle = pCtx->handle;
    unsigned char data_out[1 + len];
    data_out[0] = 0x16U;
    memcpy(data_out + 1, pData, len);

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    switch (r) {
        case 0:
            printf("SYN command sent successfully\n");
            return LW400_SYN_ERR_SUCCESS;
        default:
            fprintf(stderr, "Error sending SYN command: %s\n", libusb_error_name(r));
            return LW400_SYN_ERR_SEND_COMMAND;
    }

}

printer_err_t lw400_etb(printer_ctx_t* pCtx, uint8_t* data, uint8_t len) {
    libusb_device_handle* pHandle = pCtx->handle;
    unsigned char data_out[1 + len];
    data_out[0] = 0x17U;
    memcpy(data_out + 1, data, len);

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    switch (r) {
        case 0:
            printf("ETB command sent successfully\n");
            return LW400_ETB_ERR_SUCCESS;
        default:
            fprintf(stderr, "Error sending ETB command: %s\n", libusb_error_name(r));
            return LW400_ETB_ERR_SEND_COMMAND;
    }
}

printer_err_t lw400_esc_h(printer_ctx_t* pCtx) {
    libusb_device_handle* pHandle = pCtx->handle;
    unsigned char data_out[] = {0x1BU, 0x68U};

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    switch (r) {
        case 0:
            printf("ESC_h command sent successfully\n");
            return LW400_ESC_h_ERR_SUCCESS;
        default:
            fprintf(stderr, "Error sending ESC_h command: %s\n", libusb_error_name(r));
            return LW400_ESC_h_ERR_SEND_COMMAND;
    }

}

printer_err_t lw400_esc_i(printer_ctx_t* pCtx) {
    libusb_device_handle* pHandle = pCtx->handle;
    unsigned char data_out[] = {0x1BU, 0x69U};

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    switch (r) {
        case 0:
            printf("ESC_i command sent successfully\n");
            return LW400_ESC_i_ERR_SUCCESS;
        default:
            fprintf(stderr, "Error sending ESC_i command: %s\n", libusb_error_name(r));
            return LW400_ESC_i_ERR_SEND_COMMAND;
    }

}

printer_err_t lw400_esc_c(printer_ctx_t* pCtx) {
    libusb_device_handle* pHandle = pCtx->handle;
    unsigned char data_out[] = {0x1BU, 0x63U};

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    switch (r) {
        case 0:
            printf("ESC_c command sent successfully\n");
            return LW400_ESC_c_ERR_SUCCESS;
        default:
            fprintf(stderr, "Error sending ESC_c command: %s\n", libusb_error_name(r));
            return LW400_ESC_c_ERR_SEND_COMMAND;
    }

}

printer_err_t lw400_esc_d(printer_ctx_t* pCtx) {
    libusb_device_handle* pHandle = pCtx->handle;
    unsigned char data_out[] = {0x1BU, 0x64U};

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    switch (r) {
        case 0:
            printf("ESC_d command sent successfully\n");
            return LW400_ESC_d_ERR_SUCCESS;
        default:
            fprintf(stderr, "Error sending ESC_d command: %s\n", libusb_error_name(r));
            return LW400_ESC_d_ERR_SEND_COMMAND;
    }

}

printer_err_t lw400_esc_e(printer_ctx_t* pCtx) {
    libusb_device_handle* pHandle = pCtx->handle;
    unsigned char data_out[] = {0x1BU, 0x65U};

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    switch (r) {
        case 0:
            printf("ESC_e command sent successfully\n");
            return LW400_ESC_e_ERR_SUCCESS;
        default:
            fprintf(stderr, "Error sending ESC_e command: %s\n", libusb_error_name(r));
            return LW400_ESC_e_ERR_SEND_COMMAND;
    }

}

printer_err_t lw400_esc_g(printer_ctx_t* pCtx) {
    libusb_device_handle* pHandle = pCtx->handle;
    unsigned char data_out[] = {0x1BU, 0x67U};

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    switch (r) {
        case 0:
            printf("ESC_g command sent successfully\n");
            return LW400_ESC_g_ERR_SUCCESS;
        default:
            fprintf(stderr, "Error sending ESC_g command: %s\n", libusb_error_name(r));
            return LW400_ESC_g_ERR_SEND_COMMAND;
    }

}

printer_err_t lw400_esc_y(printer_ctx_t* pCtx) {
    libusb_device_handle* pHandle = pCtx->handle;
    unsigned char data_out[] = {0x1BU, 0x79U};

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    switch (r) {
        case 0:
            printf("ESC_y command sent successfully\n");
            return LW400_ESC_y_ERR_SUCCESS;
        default:
            fprintf(stderr, "Error sending ESC_y command: %s\n", libusb_error_name(r));
            return LW400_ESC_y_ERR_SEND_COMMAND;
    }

}

printer_err_t lw400_esc_z(printer_ctx_t* pCtx) {
    libusb_device_handle* pHandle = pCtx->handle;
    unsigned char data_out[] = {0x1BU, 0x7AU};

    int r = libusb_bulk_transfer(pHandle, DYMO_ENDPONT_ADDR_OUT, data_out, sizeof(data_out), NULL, 0);
    switch (r) {
        case 0:
            printf("ESC_z command sent successfully\n");
            return LW400_ESC_z_ERR_SUCCESS;
        default:
            fprintf(stderr, "Error sending ESC_z command: %s\n", libusb_error_name(r));
            return LW400_ESC_z_ERR_SEND_COMMAND;
    }

}

// int printer_esc_d(printer_ctx_t* pCtx, uint8_t *data, int width, int height) {return 0;
//     libusb_device_handle *handle = pCtx->handle;

//     unsigned char data_out[width * height + 8];
//     unsigned char data_in[64];
//     int actual_length;

//     // Prepare the data to send to the printer
//     data_out[0] = 0x1B;
//     data_out[1] = 0x44;
//     data_out[2] = 0x01;
//     data_out[3] = 0x02;
//     int* pWidth = (int*)(data_out + 4);
//     int* pHeight = (int*)(data_out + 8);
//     *pWidth = width;
//     *pHeight = height;

//     for (int i = 0; i < width * height; i++) {
//         data_out[12 + i] = data[i];
//     }

//     // Send the command to the printer
//     int r = libusb_bulk_transfer(handle, (1 | LIBUSB_ENDPOINT_OUT), data_out, sizeof(data_out), &actual_length, 0);
//     if (r == 0 && actual_length == sizeof(data_out)) {
//         printf("Command sent successfully\n");
//     } else {
//         fprintf(stderr, "Error sending command: %s\n", libusb_error_name(r));
//         #if PRINTER_USE_DEBUG_EXTENSIONS == 1
//         __printer_debug_errno();
//         #endif
//         return PRINTER_ESC_D_ERR_SEND_COMMAND;
//     }

//     return PRINTER_ESC_D_ERR_SUCCESS;
// }

