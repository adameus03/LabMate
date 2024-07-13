#include <stdio.h>
#include <stdlib.h>
#include "libusb.h"

#include "printer.h"

#define DYMO_VENDOR_ID 0x0922
#define DYMO_PRODUCT_ID 0x0028

int printer_test_esc_v(void) {
    libusb_device_handle *handle;
    libusb_context *context = NULL;
    int r;
    unsigned char data_out[] = {0x1B, 0x56};
    unsigned char data_in[64];
    int actual_length;

    // Initialize libusb
    r = libusb_init(&context);
    if (r < 0) {
        fprintf(stderr, "Error initializing libusb: %s\n", libusb_error_name(r));
        return 1;
    }

    // Open the device
    handle = libusb_open_device_with_vid_pid(context, DYMO_VENDOR_ID, DYMO_PRODUCT_ID);
    if (handle == NULL) {
        fprintf(stderr, "Error finding USB device\n");
        libusb_exit(context);
        return 1;
    }

    // Claim the interface (assuming interface 0)
    r = libusb_claim_interface(handle, 0);
    if (r < 0) {
        fprintf(stderr, "Error claiming interface: %s\n", libusb_error_name(r));
        libusb_close(handle);
        libusb_exit(context);
        return 1;
    }

    // Send the command to the printer
    r = libusb_bulk_transfer(handle, (1 | LIBUSB_ENDPOINT_OUT), data_out, sizeof(data_out), &actual_length, 0);
    if (r == 0 && actual_length == sizeof(data_out)) {
        printf("Command sent successfully\n");
    } else {
        fprintf(stderr, "Error sending command: %s\n", libusb_error_name(r));
        libusb_release_interface(handle, 0);
        libusb_close(handle);
        libusb_exit(context);
        return 1;
    }

    // Read the response from the printer
    r = libusb_bulk_transfer(handle, (1 | LIBUSB_ENDPOINT_IN), data_in, sizeof(data_in), &actual_length, 5000); // 5 second timeout
    if (r == 0) {
        printf("Response received:\n");
        for (int i = 0; i < actual_length; i++) {
            printf("%02X ", data_in[i]);
        }
        printf("\n");
    } else {
        fprintf(stderr, "Error reading response: %s\n", libusb_error_name(r));
    }

    // Release the interface and close the device
    libusb_release_interface(handle, 0);
    libusb_close(handle);
    libusb_exit(context);

    return 0;
}