#include <stdio.h>
#include <stdlib.h>
#include "plibsys.h"
#include <psocket.h>
#include <plibsysconfig.h>
#include <pmacros.h>
#include <ptypes.h>
#include <assert.h>
#include "uhfman.h"
#include "tag_err.h"

void main_uhfman_poll_handler(uint16_t handle, void* pUserData) {
    uhfman_tag_t tag = uhfman_tag_get(handle);
    fprintf(stdout, "Tag %u, EPC: ", tag.handle);
    for (uint32_t j = 0; j < YPDR200_X22_NTF_PARAM_EPC_LENGTH; j++) {
        fprintf(stderr, "%02X ", tag.epc[j]);
    }
    fprintf(stdout, "\n");
    return;


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


    uhfmanCtx._config.flags |= UHFMAN_CTX_CONFIG_FLAG_IS_MPOLL_BUSY; //quick hack to avoid assertion failure
    fprintf(stdout, "uhfman_multiple_polling_stop\n");
    err = uhfman_multiple_polling_stop(&uhfmanCtx);
    while (UHFMAN_ERR_SUCCESS != err) {
        LOG_W("uhfd_measure_dev: uhfman_multiple_polling_stop failed with error %d, will retry until success...", err);
        err = uhfman_multiple_polling_stop(&uhfmanCtx);
    }


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
    assert(action != UHFMAN_SELECT_ACTION_UNKNOWN);
    printf("Select action = 0x%02X\n", action);
    uint8_t memBank = UHFMAN_SELECT_MEMBANK_EPC;
    uint32_t ptr = 0x20;
    //uint8_t maskLen = 0x60;
    uint8_t maskLen = 0x00;
    uint8_t truncate = UHFMAN_SELECT_TRUNCATION_DISABLED;
    const uint8_t mask[0] = {};
    //const uint8_t mask[12] = {
        //0xE2, 0x80, 0x69, 0x15, 0x00, 0x00, 0x40, 0x17, 0xAA, 0xE6, 0x69, 0xBC
        //0xE2, 0x80, 0x69, 0x15, 0x00, 0x00, 0x40, 0x17, 0xAA, 0xE6, 0x69, 0xBD
        //0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA, 0xEA 
        //0xE2, 0x80, 0x68, 0x94, 0x00, 0x00, 0x40, 0x24, 0xED, 0x64, 0x21, 0x84 
        //0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB
    //};

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
    err = uhfman_set_query_params(&uhfmanCtx, UHFMAN_QUERY_SEL_SL, UHFMAN_QUERY_SESSION_S0, UHFMAN_QUERY_TARGET_A, 0x04);
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


    fprintf(stdout, "!!! Calling uhfman_set_transmit_power !!!\n");
    err = uhfman_set_transmit_power(&uhfmanCtx, 25.0f);
    if (err != UHFMAN_SET_TRANSMIT_POWER_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_set_transmit_power returned %d\n", err);
        //return 1;
    } else {
        fprintf(stdout, "uhfman_set_transmit_power returned successfully\n");
    }


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

    fprintf(stdout, "Calling uhfman_set_poll_handler\n");
    uhfman_set_poll_handler(main_uhfman_poll_handler);

    // fprintf(stdout, "Calling uhfman_dbg_single_polling\n");
    // err = uhfman_single_polling(&uhfmanCtx, NULL);
    // if (err != UHFMAN_SINGLE_POLLING_ERR_SUCCESS) {
    //     P_ERROR("USB related error"); // TODO improve those error messages, theses are not really neccessarily USB related, but rather related to underlying UHF RFID interrogator module
    //     fprintf(stderr, "ERROR (ignoring): uhfman_dbg_single_polling returned %d\n", err);
    //     //return 1;
    // } else {
    //     fprintf(stdout, "uhfman_dbg_single_polling returned successfully\n");
    // }

    fprintf(stdout, "Calling uhfman_dbg_multiple_polling\n");
    err = uhfman_multiple_polling(&uhfmanCtx, 5000000, NULL);
    if (err != UHFMAN_MULTIPLE_POLLING_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_dbg_multiple_polling returned %d\n", err);
        //return 1;
    } else {
        fprintf(stdout, "uhfman_dbg_multiple_polling returned successfully\n");
    }

    err = uhfman_multiple_polling_stop(&uhfmanCtx);
    int try_counter = 0;
    while (UHFMAN_ERR_SUCCESS != err) {
        try_counter++;
        if (try_counter == 10) {
            fprintf(stderr, "uhfman_multiple_polling_stop failed 10 times, exiting...\n");
            uhfman_unset_poll_handler();

            fprintf(stdout, "Calling uhfman_device_release after failure\n");
            uhfman_device_release(&uhfmanCtx);
            fprintf(stdout, "uhfman_device_release returned after failure\n");
            return 1;
        }
        LOG_W("uhfd_measure_dev: uhfman_multiple_polling_stop failed with error %d, will retry until success...", err);
        err = uhfman_multiple_polling_stop(&uhfmanCtx);
    }
    fprintf(stdout, "uhfman_multiple_polling_stop returned successfully. Unsetting poll handler\n");
    uhfman_unset_poll_handler();

    fprintf(stdout, "Calling uhfman_device_release\n");
    uhfman_device_release(&uhfmanCtx);
    fprintf(stdout, "uhfman_device_release returned\n");

    return 0;
}
