#include <plibsys/plibsys.h>
#include "log.h"
#include "uhfman.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Structure to hold measurement data
typedef struct {
    uint8_t epc[YPDR200_X22_NTF_PARAM_EPC_LENGTH];
    time_t timestamp;
    uint8_t rssi; // Updated to use actual RSSI value
} measurement_t;

// Structure to hold experiment results
typedef struct {
    // Device information
    char* hardware_version;
    char* software_version;
    char* manufacturer;
    
    // Requested configuration parameters
    struct {
        uint8_t select_target;
        uint8_t select_action;
        uint8_t select_membank;
        uint32_t select_ptr;
        uint8_t select_mask_len;
        uint8_t select_truncate;
        uint8_t query_sel;
        uint8_t query_session;
        uint8_t query_target;
        uint8_t query_q;
        float transmit_power;
        uint8_t select_mode;
    } requested_config;
    
    // Actual configuration parameters (populated from getter functions)
    struct {
        uint8_t actual_select_target;
        uint8_t actual_select_action;
        uint8_t actual_select_membank;
        uint32_t actual_select_ptr;
        uint8_t actual_select_mask_len;
        uint8_t actual_select_truncate;
        uint8_t* actual_select_mask;
        uint8_t actual_query_sel;
        uint8_t actual_query_session;
        uint8_t actual_query_target;
        uint8_t actual_query_q;
        float actual_transmit_power;
        uint8_t actual_select_mode_placeholder; // No getter available yet
        uint8_t actual_working_channel;
        int actual_work_area;
        int actual_mixer_gain;
        int actual_if_gain;
        uint16_t actual_threshold;
    } actual_config;
    
    // Measurement data
    measurement_t* measurements;
    size_t measurement_count;
    size_t measurement_capacity;
    
    // Experiment metadata
    time_t start_time;
    time_t end_time;
    uint32_t polling_duration_ms;
    int success;
} experiment_result_t;

// Global experiment result
static experiment_result_t g_experiment_result = {0};

void init_experiment_result(experiment_result_t* result) {
    memset(result, 0, sizeof(experiment_result_t));
    result->measurement_capacity = 1000; // Initial capacity
    result->measurements = malloc(sizeof(measurement_t) * result->measurement_capacity);
    result->start_time = time(NULL);
}

void add_measurement(experiment_result_t* result, const uhfman_tag_t* tag, uint8_t rssi) {
    if (result->measurement_count >= result->measurement_capacity) {
        result->measurement_capacity *= 2;
        result->measurements = realloc(result->measurements, sizeof(measurement_t) * result->measurement_capacity);
    }
    
    measurement_t* measurement = &result->measurements[result->measurement_count];
    memcpy(measurement->epc, tag->epc, YPDR200_X22_NTF_PARAM_EPC_LENGTH);
    measurement->timestamp = time(NULL);
    measurement->rssi = rssi;
    
    result->measurement_count++;
}

void main_uhfman_poll_handler(uhfman_tag_t tagInfo, uint8_t rssi, void* pUserData) {
    add_measurement(&g_experiment_result, &tagInfo, rssi);
    fprintf(stdout, "Tag detected: RSSI=%u, EPC=", rssi);
    for (int i = 0; i < YPDR200_X22_NTF_PARAM_EPC_LENGTH; i++) {
        fprintf(stdout, "%02X", tagInfo.epc[i]);
    }
    fprintf(stdout, "\n");
}

char* escape_json_string(const char* str) {
    if (!str) return strdup("null");
    
    size_t len = strlen(str);
    char* escaped = malloc(len * 2 + 3); // Worst case: every char needs escaping + quotes + null
    char* dest = escaped;
    
    *dest++ = '"';
    for (const char* src = str; *src; src++) {
        switch (*src) {
            case '"': *dest++ = '\\'; *dest++ = '"'; break;
            case '\\': *dest++ = '\\'; *dest++ = '\\'; break;
            case '\n': *dest++ = '\\'; *dest++ = 'n'; break;
            case '\r': *dest++ = '\\'; *dest++ = 'r'; break;
            case '\t': *dest++ = '\\'; *dest++ = 't'; break;
            default: *dest++ = *src; break;
        }
    }
    *dest++ = '"';
    *dest = '\0';
    
    return escaped;
}

char* generate_json_result(const experiment_result_t* result) {
    char* json = malloc(65536); // Start with 64KB buffer
    size_t json_size = 65536;
    size_t json_len = 0;
    
    // Helper macro to append to JSON string
    #define APPEND_JSON(fmt, ...) do { \
        int needed = snprintf(json + json_len, json_size - json_len, fmt, ##__VA_ARGS__); \
        if (needed >= (int)(json_size - json_len)) { \
            json_size = json_size * 2 + needed; \
            json = realloc(json, json_size); \
            snprintf(json + json_len, json_size - json_len, fmt, ##__VA_ARGS__); \
        } \
        json_len += needed; \
    } while(0)
    
    APPEND_JSON("{\n");
    
    // Device information
    APPEND_JSON("  \"device_info\": {\n");
    char* hw_escaped = escape_json_string(result->hardware_version);
    char* sw_escaped = escape_json_string(result->software_version);
    char* mfg_escaped = escape_json_string(result->manufacturer);
    APPEND_JSON("    \"hardware_version\": %s,\n", hw_escaped);
    APPEND_JSON("    \"software_version\": %s,\n", sw_escaped);
    APPEND_JSON("    \"manufacturer\": %s\n", mfg_escaped);
    free(hw_escaped);
    free(sw_escaped);
    free(mfg_escaped);
    APPEND_JSON("  },\n");
    
    // Requested configuration
    APPEND_JSON("  \"requested_config\": {\n");
    APPEND_JSON("    \"select_target\": %u,\n", result->requested_config.select_target);
    APPEND_JSON("    \"select_action\": %u,\n", result->requested_config.select_action);
    APPEND_JSON("    \"select_membank\": %u,\n", result->requested_config.select_membank);
    APPEND_JSON("    \"select_ptr\": %u,\n", result->requested_config.select_ptr);
    APPEND_JSON("    \"select_mask_len\": %u,\n", result->requested_config.select_mask_len);
    APPEND_JSON("    \"select_truncate\": %u,\n", result->requested_config.select_truncate);
    APPEND_JSON("    \"query_sel\": %u,\n", result->requested_config.query_sel);
    APPEND_JSON("    \"query_session\": %u,\n", result->requested_config.query_session);
    APPEND_JSON("    \"query_target\": %u,\n", result->requested_config.query_target);
    APPEND_JSON("    \"query_q\": %u,\n", result->requested_config.query_q);
    APPEND_JSON("    \"transmit_power\": %.1f,\n", result->requested_config.transmit_power);
    APPEND_JSON("    \"select_mode\": %u\n", result->requested_config.select_mode);
    APPEND_JSON("  },\n");
    
    // Actual configuration
    APPEND_JSON("  \"actual_config\": {\n");
    APPEND_JSON("    \"actual_select_target\": %u,\n", result->actual_config.actual_select_target);
    APPEND_JSON("    \"actual_select_action\": %u,\n", result->actual_config.actual_select_action);
    APPEND_JSON("    \"actual_select_membank\": %u,\n", result->actual_config.actual_select_membank);
    APPEND_JSON("    \"actual_select_ptr\": %u,\n", result->actual_config.actual_select_ptr);
    APPEND_JSON("    \"actual_select_mask_len\": %u,\n", result->actual_config.actual_select_mask_len);
    APPEND_JSON("    \"actual_select_truncate\": %u,\n", result->actual_config.actual_select_truncate);
    
    // Handle select mask array
    APPEND_JSON("    \"actual_select_mask\": \"");
    if (result->actual_config.actual_select_mask && result->actual_config.actual_select_mask_len > 0) {
        for (uint8_t i = 0; i < result->actual_config.actual_select_mask_len; i++) {
            APPEND_JSON("%02X", result->actual_config.actual_select_mask[i]);
        }
    }
    APPEND_JSON("\",\n");
    
    APPEND_JSON("    \"actual_query_sel\": %u,\n", result->actual_config.actual_query_sel);
    APPEND_JSON("    \"actual_query_session\": %u,\n", result->actual_config.actual_query_session);
    APPEND_JSON("    \"actual_query_target\": %u,\n", result->actual_config.actual_query_target);
    APPEND_JSON("    \"actual_query_q\": %u,\n", result->actual_config.actual_query_q);
    APPEND_JSON("    \"actual_transmit_power\": %.1f,\n", result->actual_config.actual_transmit_power);
    APPEND_JSON("    \"actual_select_mode_placeholder\": %u,\n", result->actual_config.actual_select_mode_placeholder);
    APPEND_JSON("    \"actual_working_channel\": %u,\n", result->actual_config.actual_working_channel);
    APPEND_JSON("    \"actual_work_area\": %d,\n", result->actual_config.actual_work_area);
    APPEND_JSON("    \"actual_mixer_gain\": %d,\n", result->actual_config.actual_mixer_gain);
    APPEND_JSON("    \"actual_if_gain\": %d,\n", result->actual_config.actual_if_gain);
    APPEND_JSON("    \"actual_threshold\": %u\n", result->actual_config.actual_threshold);
    APPEND_JSON("  },\n");
    
    // Experiment metadata
    APPEND_JSON("  \"experiment_metadata\": {\n");
    APPEND_JSON("    \"start_time\": %ld,\n", result->start_time);
    APPEND_JSON("    \"end_time\": %ld,\n", result->end_time);
    APPEND_JSON("    \"polling_duration_ms\": %u,\n", result->polling_duration_ms);
    APPEND_JSON("    \"success\": %s,\n", result->success ? "true" : "false");
    APPEND_JSON("    \"measurement_count\": %zu\n", result->measurement_count);
    APPEND_JSON("  },\n");
    
    // Measurements array
    APPEND_JSON("  \"measurements\": [\n");
    for (size_t i = 0; i < result->measurement_count; i++) {
        const measurement_t* m = &result->measurements[i];
        APPEND_JSON("    {\n");
        APPEND_JSON("      \"timestamp\": %ld,\n", m->timestamp);
        APPEND_JSON("      \"epc\": \"");
        for (uint32_t j = 0; j < YPDR200_X22_NTF_PARAM_EPC_LENGTH; j++) {
            APPEND_JSON("%02X", m->epc[j]);
        }
        APPEND_JSON("\",\n");
        APPEND_JSON("      \"rssi\": %u\n", m->rssi);
        APPEND_JSON("    }%s\n", (i < result->measurement_count - 1) ? "," : "");
    }
    APPEND_JSON("  ]\n");
    APPEND_JSON("}\n");
    
    #undef APPEND_JSON
    
    return json;
}

void cleanup_experiment_result(experiment_result_t* result) {
    if (result->measurements) {
        free(result->measurements);
        result->measurements = NULL;
    }
    if (result->hardware_version) {
        free(result->hardware_version);
        result->hardware_version = NULL;
    }
    if (result->software_version) {
        free(result->software_version);
        result->software_version = NULL;
    }
    if (result->manufacturer) {
        free(result->manufacturer);
        result->manufacturer = NULL;
    }
    if (result->actual_config.actual_select_mask) {
        free(result->actual_config.actual_select_mask);
        result->actual_config.actual_select_mask = NULL;
    }
}

int main_rssi_experiment(int polling_duration_ms) {
    uhfman_ctx_t uhfmanCtx = {};
    uhfman_err_t err;
    
    init_experiment_result(&g_experiment_result);
    
    fprintf(stdout, "Calling uhfman_device_take\n");
    err = uhfman_device_take(&uhfmanCtx);
    if (err != UHFMAN_TAKE_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "uhfman_device_take returned %d\n", err);
        g_experiment_result.success = 0;
        return 1;
    }
    fprintf(stdout, "uhfman_device_take returned successfully\n");

    uhfmanCtx._config.flags |= UHFMAN_CTX_CONFIG_FLAG_IS_MPOLL_BUSY;
    fprintf(stdout, "uhfman_multiple_polling_stop\n");
    err = uhfman_multiple_polling_stop(&uhfmanCtx);
    while (UHFMAN_ERR_SUCCESS != err) {
        LOG_W("uhfd_measure_dev: uhfman_multiple_polling_stop failed with error %d, will retry until success...", err);
        err = uhfman_multiple_polling_stop(&uhfmanCtx);
    }

    // Get device information
    fprintf(stdout, "Calling uhfman_get_hardware_version\n");
    err = uhfman_get_hardware_version(&uhfmanCtx, &g_experiment_result.hardware_version);
    if (err != UHFMAN_GET_HARDWARE_VERSION_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_get_hardware_version returned %d\n", err);
    }

    fprintf(stdout, "Calling uhfman_get_software_version\n");
    err = uhfman_get_software_version(&uhfmanCtx, &g_experiment_result.software_version);
    if (err != UHFMAN_GET_SOFTWARE_VERSION_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_get_software_version returned %d\n", err);
    }

    fprintf(stdout, "Calling uhfman_get_manufacturer\n");
    err = uhfman_get_manufacturer(&uhfmanCtx, &g_experiment_result.manufacturer);
    if (err != UHFMAN_GET_MANUFACTURER_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_get_manufacturer returned %d\n", err);
    }

    // Set and store configuration parameters
    fprintf(stdout, "Calling uhfman_set_select_param\n");
    uint8_t target = UHFMAN_SELECT_TARGET_SL;
    uint8_t action = uhfman_select_action(UHFMAN_SEL_SL_ASSERT, UHFMAN_SEL_SL_DEASSERT);
    assert(action != UHFMAN_SELECT_ACTION_UNKNOWN);
    uint8_t memBank = UHFMAN_SELECT_MEMBANK_EPC;
    uint32_t ptr = 0x20;
    uint8_t maskLen = 0x00;
    uint8_t truncate = UHFMAN_SELECT_TRUNCATION_DISABLED;
    const uint8_t mask[0] = {};

    // Store requested configuration
    g_experiment_result.requested_config.select_target = target;
    g_experiment_result.requested_config.select_action = action;
    g_experiment_result.requested_config.select_membank = memBank;
    g_experiment_result.requested_config.select_ptr = ptr;
    g_experiment_result.requested_config.select_mask_len = maskLen;
    g_experiment_result.requested_config.select_truncate = truncate;

    err = uhfman_set_select_param(&uhfmanCtx, target, action, memBank, ptr, maskLen, truncate, mask);
    if (err != UHFMAN_SET_SELECT_PARAM_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_set_select_param returned %d\n", err);
    }

    // Get actual select parameters
    fprintf(stdout, "Calling uhfman_get_select_param\n");
    err = uhfman_get_select_param(&uhfmanCtx, 
                                  &g_experiment_result.actual_config.actual_select_target,
                                  &g_experiment_result.actual_config.actual_select_action,
                                  &g_experiment_result.actual_config.actual_select_membank,
                                  &g_experiment_result.actual_config.actual_select_ptr,
                                  &g_experiment_result.actual_config.actual_select_mask_len,
                                  &g_experiment_result.actual_config.actual_select_truncate,
                                  &g_experiment_result.actual_config.actual_select_mask);
    if (err != UHFMAN_GET_SELECT_PARAM_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_get_select_param returned %d\n", err);
    }

    fprintf(stdout, "!!! Calling uhfman_set_query_params !!!\n");
    uint8_t query_sel = UHFMAN_QUERY_SEL_SL;
    uint8_t query_session = UHFMAN_QUERY_SESSION_S0;
    uint8_t query_target = UHFMAN_QUERY_TARGET_A;
    uint8_t query_q = 0x04;
    
    // Store requested query parameters
    g_experiment_result.requested_config.query_sel = query_sel;
    g_experiment_result.requested_config.query_session = query_session;
    g_experiment_result.requested_config.query_target = query_target;
    g_experiment_result.requested_config.query_q = query_q;
    
    err = uhfman_set_query_params(&uhfmanCtx, query_sel, query_session, query_target, query_q);
    if (err != UHFMAN_SET_QUERY_PARAMS_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_set_query_params returned %d\n", err);
    }

    // Get actual query parameters
    fprintf(stdout, "Calling uhfman_get_query_params\n");
    err = uhfman_get_query_params(&uhfmanCtx,
                                  (uhfman_query_sel_t*)&g_experiment_result.actual_config.actual_query_sel,
                                  (uhfman_query_session_t*)&g_experiment_result.actual_config.actual_query_session,
                                  (uhfman_query_target_t*)&g_experiment_result.actual_config.actual_query_target,
                                  &g_experiment_result.actual_config.actual_query_q);
    if (err != UHFMAN_GET_QUERY_PARAMS_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_get_query_params returned %d\n", err);
    }

    // Get working channel
    fprintf(stdout, "Calling uhfman_get_working_channel\n");
    err = uhfman_get_working_channel(&uhfmanCtx, &g_experiment_result.actual_config.actual_working_channel);
    if (err != UHFMAN_GET_WORKING_CHANNEL_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_get_working_channel returned %d\n", err);
    }

    // Get work area
    fprintf(stdout, "Calling uhfman_get_work_area\n");
    err = uhfman_get_work_area(&uhfmanCtx, &g_experiment_result.actual_config.actual_work_area);
    if (err != UHFMAN_GET_WORK_AREA_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_get_work_area returned %d\n", err);
    }

    fprintf(stdout, "!!! Calling uhfman_set_transmit_power !!!\n");
    float transmit_power = 25.0f;
    g_experiment_result.requested_config.transmit_power = transmit_power;
    
    err = uhfman_set_transmit_power(&uhfmanCtx, transmit_power);
    if (err != UHFMAN_SET_TRANSMIT_POWER_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_set_transmit_power returned %d\n", err);
    }

    // Get actual transmit power
    fprintf(stdout, "Calling uhfman_get_transmit_power\n");
    err = uhfman_get_transmit_power(&uhfmanCtx, &g_experiment_result.actual_config.actual_transmit_power);
    if (err != UHFMAN_GET_TRANSMIT_POWER_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_get_transmit_power returned %d\n", err);
    }

    // Get demod parameters
    fprintf(stdout, "Calling uhfman_get_demod_params\n");
    err = uhfman_get_demod_params(&uhfmanCtx,
                                  &g_experiment_result.actual_config.actual_mixer_gain,
                                  &g_experiment_result.actual_config.actual_if_gain,
                                  &g_experiment_result.actual_config.actual_threshold);
    if (err != UHFMAN_GET_DEMOD_PARAMS_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_get_demod_params returned %d\n", err);
    }

    fprintf(stdout, "Calling uhfman_set_select_mode\n");
    uint8_t select_mode = UHFMAN_SELECT_MODE_ALWAYS;
    g_experiment_result.requested_config.select_mode = select_mode;
    
    err = uhfman_set_select_mode(&uhfmanCtx, select_mode);
    if (err != UHFMAN_SET_SELECT_MODE_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_set_select_mode returned %d\n", err);
    }

    // Note: No getter for select_mode available, keeping as placeholder
    g_experiment_result.actual_config.actual_select_mode_placeholder = 0;

    fprintf(stdout, "Calling uhfman_set_poll_handler\n");
    uhfman_set_poll_handler(main_uhfman_poll_handler);

    fprintf(stdout, "Calling uhfman_multiple_polling\n");
    //uint32_t polling_duration = 5000000; // 5 seconds in microseconds
    uint32_t polling_duration = polling_duration_ms * 1000; // Convert ms to us
    g_experiment_result.polling_duration_ms = polling_duration / 1000; // Convert to milliseconds for JSON
    
    err = uhfman_multiple_polling(&uhfmanCtx, polling_duration, NULL);
    if (err != UHFMAN_MULTIPLE_POLLING_ERR_SUCCESS) {
        P_ERROR("USB related error");
        fprintf(stderr, "ERROR (ignoring): uhfman_multiple_polling returned %d\n", err);
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
            g_experiment_result.success = 0;
            g_experiment_result.end_time = time(NULL);
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

    g_experiment_result.success = 1;
    g_experiment_result.end_time = time(NULL);
    
    return 0;
}

int main(int argc, char **argv) {
    p_libsys_init();
    log_global_init();
    
    const char* output_json_file_path = argc > 1 ? argv[1] : "rssi_stats_ll_result.json";
    int polling_duration_ms = argc > 2 ? atoi(argv[2]) : 5000; // Default to 5000 ms if not provided

    int result = main_rssi_experiment(polling_duration_ms);
    
    // Generate and output JSON
    char* json_output = generate_json_result(&g_experiment_result);
    // fprintf(stdout, "\n=== EXPERIMENT RESULT JSON ===\n");
    // fprintf(stdout, "%s", json_output);
    // fprintf(stdout, "=== END EXPERIMENT RESULT JSON ===\n");

    FILE* f = fopen(output_json_file_path, "w");
    if (f) {
      fprintf(f, "%s", json_output);
      fclose(f);
      fprintf(stdout, "Experiment result written to rssi_stats_ll_result.json\n");
    } else {
      fprintf(stderr, "Failed to open rssi_stats_ll_result.json for writing\n");
    }
    
    // Cleanup
    free(json_output);
    cleanup_experiment_result(&g_experiment_result);
    
    log_global_deinit();
    p_libsys_shutdown();
    return result;
}