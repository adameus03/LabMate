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
        fprintf(stderr, "uhfman_get_hardware_version returned %d\n", err);
        return 1;
    }
    if (hardwareVersion == NULL) {
        fprintf(stderr, "Hardware version is NULL\n");
    } else {
        fprintf(stdout, "Hardware version: %s\n", hardwareVersion);
    }

    fprintf(stdout, "Calling uhfman_device_release\n");
    uhfman_device_release(&uhfmanCtx);
    fprintf(stdout, "uhfman_device_release returned\n");

    return 0;
}