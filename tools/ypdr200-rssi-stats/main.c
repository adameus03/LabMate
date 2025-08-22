#include <plibsys/plibsys.h>
#include <stdlib.h>
#include "config.h"
#include "log.h"
#include "acall.h"
#include "rscall.h"

#define TARGET_ANTENNA 1
#define TARGET_TAG_EPC "bbbbbbbbbbbbbbbbbbbbbbbb"
#define TX_POWER 26

int main_configure_target_antenna(void) {
  const char* target_antenna_path = acall_ant_get_path(TARGET_ANTENNA);
  if (target_antenna_path == NULL) {
    LOG_E("main_rssi_experiment: acall_ant_get_path failed for index %d", TARGET_ANTENNA);
    return -1;
  }
  int rv = acall_ant_set_enabled(target_antenna_path);
  if (rv != 0) {
    LOG_E("main_rssi_experiment: acall_ant_set_enabled failed for path %s with rv=%d", target_antenna_path, rv);
    free((void*)target_antenna_path);
    return -2;
  }
  LOG_I("main_rssi_experiment: Enabled antenna %d (path: %s)", TARGET_ANTENNA, target_antenna_path);
  free((void*)target_antenna_path); // free the path after use
  return 0;
}

/**
 * @note Output buffer needs to be freed with free() after use
 */
const char* main_prepare_target_tag(void) {
  const char* iePath = rscall_ie_dir_create();
  if (iePath == NULL) {
    LOG_E("main_prepare_target_tag: rscall_ie_dir_create failed");
    return NULL;
  }
  int rv = rscall_ie_set_epc(iePath, TARGET_TAG_EPC);
  if (rv != 0) {
    LOG_E("main_prepare_target_tag: rscall_ie_set_epc failed for path %s with rv=%d", iePath, rv);
    free((void*)iePath);
    return NULL;
  }
  rv = rscall_ie_set_flags(iePath, "02"); // Inform the driver that the tag was already its EPC written
  if (rv != 0) {
    LOG_E("main_prepare_target_tag: rscall_ie_set_flags failed for path %s with rv=%d", iePath, rv);
    free((void*)iePath);
    return NULL;
  }
  return iePath;
}

#define USER_ACTION_UNDEFINED 0
#define USER_ACTION_QUIT 1
#define USER_ACTION_MEASURE 2

// If user pressed 'q' - exit
// If user pressed 'm' - perform a measurement
// Else - ignore
int main_get_user_action(void) {
  int c = getchar();
  if (c == 'q') {
    return USER_ACTION_QUIT; // quit
  } else if (c == 'm') {
    return USER_ACTION_MEASURE; // measure
  }
  return USER_ACTION_UNDEFINED; // ignore
}

void main_rssi_experiment(void) {
  LOG_I("Beginning RSSI experiment");
  // Configure antenna
  LOG_I("main_rssi_experiment: Configuring target antenna %d", TARGET_ANTENNA);
  int rv = main_configure_target_antenna();
  if (rv != 0) {
    LOG_E("main_rssi_experiment: main_configure_target_antenna failed with rv=%d", rv);
    return;
  }

  // Prepare target tag
  LOG_I("main_rssi_experiment: Preparing target tag with EPC %s", TARGET_TAG_EPC);
  const char* target_ie_path = main_prepare_target_tag();
  if (target_ie_path == NULL) {
    LOG_E("main_rssi_experiment: main_prepare_target_tag failed");
    return;
  }

  // Perform measurements
  for (int i = 0; 1; i++) {
    int user_action = main_get_user_action();
    if (user_action == USER_ACTION_QUIT) {
      LOG_I("main_rssi_experiment: User requested to quit the experiment");
      break;
    }

    LOG_I("main_rssi_experiment: Performing measurement iteration %d", i);
    rv = rscall_ie_drv_measure_quick(target_ie_path, TX_POWER);
    if (rv != 0) {
      LOG_E("main_rssi_experiment: rscall_ie_drv_measure_quick failed for path %s with rv=%d", target_ie_path, rv);
      continue;
    }
    int rssi = 0;
    rv = rscall_ie_get_rssi(target_ie_path, &rssi);
    if (rv != 0) {
      LOG_E("main_rssi_experiment: rscall_ie_get_rssi failed for path %s with rv=%d", target_ie_path, rv);
      continue;
    }
    LOG_I("main_rssi_experiment: Measured RSSI=%d for path %s", rssi, target_ie_path);
    p_uthread_sleep(1000); // Sleep 1 second between measurements
  }
  //Cleanup
  free((void*)target_ie_path); // free the path after use

  LOG_I("main_rssi_experiment: All measurements completed");
}

int main(int argc, char **argv) {
  p_libsys_init();
  log_global_init();
  main_rssi_experiment();
  log_global_deinit();
  p_libsys_shutdown();
  return 0;
}