#include "rscall.h"
#include "config.h"
#include "log.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <plibsys/plibsys.h>

#ifndef RSCALL_RSCS_MOUNT_PATH
    #error "RSCALL_RSCS_MOUNT_PATH is not defined!"
#endif

/**
 * @warning The returned path needs to be freed with free()
 */
static char* __rscall_abspath(const char* __relpath) {
  const char* mountPath = RSCALL_RSCS_MOUNT_PATH;
  size_t mountPathStrlen = strlen(mountPath);
  size_t absPathStrlen = mountPathStrlen + strlen(__relpath);
  if (mountPath[mountPathStrlen - 1] != '/') {
    absPathStrlen++;
  }
  const char* relpath = __relpath;
  if (relpath[0] == '/') {
    relpath++;
  }
  char* absPath = (char*)malloc(absPathStrlen + 1);
  assert(absPath != NULL);
  strcpy(absPath, mountPath);
  if (mountPath[mountPathStrlen - 1] != '/') {
      absPath[mountPathStrlen] = '/';
      strcpy(absPath + mountPathStrlen + 1, relpath);
  } else {
      strcpy(absPath + mountPathStrlen, relpath);
  }
  return absPath;
}

char* rscall_ie_dir_create(void) {
  const char* sid_path = __rscall_abspath("uhfd/sid");
  const char* mkdev_path = __rscall_abspath("uhfd/mkdev");
  
  // Read from uhfd/sid to get the SID
  FILE* sid_file = fopen(sid_path, "r");
  if (sid_file == NULL) {
    free((void*)sid_path);
    free((void*)mkdev_path);
    return NULL;
  }
  char sid[129]; // We should not exceed 128 characters in SID as all we read is a natural number
  if (fgets(sid, sizeof(sid), sid_file) == NULL) {
    fclose(sid_file);
    free((void*)sid_path);
    free((void*)mkdev_path);
    return NULL;
  }
  size_t sidStrlen = strlen(sid);
  assert(sidStrlen > 0 && sidStrlen < 128); // If we have 128 characters, there is something going on wrong definitely
  fclose(sid_file);

  // Write to uhfd/mkdev to trigger the creation of a new device
  FILE* mkdev_file = fopen(mkdev_path, "w");
  if (mkdev_file == NULL) {
    free((void*)sid_path);
    free((void*)mkdev_path);
    assert(0);
    return NULL;
  }
  if (fprintf(mkdev_file, "%s", sid) <= 0) {
    free((void*)sid_path);
    free((void*)mkdev_path);
    assert(0 == fclose(mkdev_file));
    assert(0);
    return NULL;
  }
  assert(0 == fclose(mkdev_file));

  // Read from result/$sid/value to get the devno
  char* devno_path_prolog = __rscall_abspath("uhfd/result/");
  size_t devno_path_prolog_strlen = strlen(devno_path_prolog);
  char* devno_path_epilog = "/value";
  size_t devno_path_epilog_strlen = strlen(devno_path_epilog);
  char* devno_path = (char*)malloc(devno_path_prolog_strlen + sidStrlen + devno_path_epilog_strlen + 1);
  assert(devno_path != NULL);
  strcpy(devno_path, devno_path_prolog);
  strcpy(devno_path + devno_path_prolog_strlen, sid);
  strcpy(devno_path + devno_path_prolog_strlen + sidStrlen, devno_path_epilog);
  devno_path[devno_path_prolog_strlen + sidStrlen + devno_path_epilog_strlen] = '\0';
  FILE* devno_file = fopen(devno_path, "r");
  if (devno_file == NULL) {
    free((void*)sid_path);
    free((void*)mkdev_path);
    free((void*)devno_path_prolog);
    free((void*)devno_path);
    assert(0);
    return NULL;
  }
  char devno[129]; // We should not exceed 128 characters in devno as all we read is a natural number
  if (fgets(devno, sizeof(devno), devno_file) == NULL) {
    fclose(devno_file);
    free((void*)sid_path);
    free((void*)mkdev_path);
    free((void*)devno_path_prolog);
    free((void*)devno_path);
    return NULL;
  }
  size_t devnoStrlen = strlen(devno);
  assert(devnoStrlen > 0 && devnoStrlen < 128); // If we have 128 characters, there is something going on wrong definitely
  fclose(devno_file);

  // Construct the path to the new device directory
  char* ie_dir_path_prolog = __rscall_abspath("uhf");
  size_t ie_dir_path_prolog_strlen = strlen(ie_dir_path_prolog);
  assert(ie_dir_path_prolog[ie_dir_path_prolog_strlen - 1] != '/');
  char* ie_dir_path = (char*)malloc(ie_dir_path_prolog_strlen + devnoStrlen + 1);
  assert(ie_dir_path != NULL);
  strcpy(ie_dir_path, ie_dir_path_prolog);
  strcpy(ie_dir_path + ie_dir_path_prolog_strlen, devno);
  ie_dir_path[ie_dir_path_prolog_strlen + devnoStrlen] = '\0';

  free((void*)sid_path);
  free((void*)mkdev_path);
  free((void*)devno_path_prolog);
  free((void*)devno_path);
  return ie_dir_path;
}

// @warning `xPathEpilog` should start with '/'
static int __rscall_ie_set_x(const char* iePath, const char* xPathEpilog, const char* xValue) {
  const size_t iePathStrlen = strlen(iePath);
  assert(iePath[iePathStrlen - 1] != '/');
  const size_t x_path_epilog_strlen = strlen(xPathEpilog);
  assert(xPathEpilog[0] == '/');
  char* x_path = (char*)malloc(iePathStrlen + x_path_epilog_strlen + 1);
  assert(x_path != NULL);
  strcpy(x_path, iePath);
  strcpy(x_path + iePathStrlen, xPathEpilog);
  x_path[iePathStrlen + x_path_epilog_strlen] = '\0';

  // Write to x
  FILE* x_file = fopen(x_path, "w");
  if (x_file == NULL) {
    //assert(0);
    return -1;
  }
  if (fprintf(x_file, "%s", xValue) <= 0) {
    assert(0 == fclose(x_file));
    assert(0);
    return -2;
  }
  int rv = fclose(x_file);
  if (rv != 0) {
    LOG_E("__rsccall_ie_set_x: fclose (path: %s) failed with rv=%d, errno=%d", x_path, rv, errno);
    return -3;
  }
  return 0;
}

int rscall_ie_set_access_passwd(const char* iePath, const char* passwd) {
  // const size_t iePathStrlen = strlen(iePath);
  // assert(iePath[iePathStrlen - 1] != '/');
  // const char* access_passwd_path_epilog = "/access_passwd";
  // const size_t access_passwd_path_epilog_strlen = strlen(access_passwd_path_epilog);
  // char* access_passwd_path = (char*)malloc(iePathStrlen + access_passwd_path_epilog_strlen + 1);
  // strcpy(access_passwd_path, iePath);
  // strcpy(access_passwd_path + iePathStrlen, access_passwd_path_epilog);
  // access_passwd_path[iePathStrlen + access_passwd_path_epilog_strlen] = '\0';

  // // Write to access_passwd
  // FILE* access_passwd_file = fopen(access_passwd_path, "w");
  // if (access_passwd_file == NULL) {
  //   assert(0);
  //   return -1;
  // }
  // if (fprintf(access_passwd_file, "%s", passwd) <= 0) {
  //   assert(0 == fclose(access_passwd_file));
  //   assert(0);
  //   return -1;
  // }
  // assert(0 == fclose(access_passwd_file));
  // return 0;
  return __rscall_ie_set_x(iePath, "/access_passwd", passwd);
}

int rscall_ie_set_kill_passwd(const char* iePath, const char* passwd) {
  return __rscall_ie_set_x(iePath, "/kill_passwd", passwd);
}

int rscall_ie_set_epc(const char* iePath, const char* epc) {
  return __rscall_ie_set_x(iePath, "/epc", epc);
}

int rscall_ie_get_epc(const char* iePath, char** ppEpc) {
  // Read from `epc`
  const size_t iePathStrlen = strlen(iePath);
  const char* epc_path_epilog = "/epc";
  const size_t epc_path_epilog_strlen = strlen(epc_path_epilog);
  char* epc_path = (char*)malloc(iePathStrlen + epc_path_epilog_strlen + 1);
  assert(epc_path != NULL);
  strcpy(epc_path, iePath);
  strcpy(epc_path + iePathStrlen, epc_path_epilog);
  epc_path[iePathStrlen + epc_path_epilog_strlen] = '\0';

  FILE* epc_file = fopen(epc_path, "r");
  if (epc_file == NULL) {
    free((void*)epc_path);
    return -1;
  }
  //TODO adjust buffer size?
  char epc[129]; // We should not exceed 128 characters in epc as all we read is a natural number
  if (fgets(epc, sizeof(epc), epc_file) == NULL) {
    fclose(epc_file);
    free((void*)epc_path);
    return -2;
  }
  size_t epcStrlen = strlen(epc);
  assert(epcStrlen > 0 && epcStrlen < 128); // If we have 128 characters, there is something going on wrong definitely
  fclose(epc_file);

  *ppEpc = p_strdup(epc);
  free((void*)epc_path);
  return 0;
}

int rscall_ie_set_flags(const char* iePath, const char* flags) {
  return __rscall_ie_set_x(iePath, "/flags", flags);
}

int rscall_ie_drv_embody(const char* iePath) {
  return __rscall_ie_set_x(iePath, "/driver/embody", "1");
}

int rscall_ie_drv_measure_quick(const char* iePath, const int txPower) {
  // To trigger measurement we write to driver/measure
  char* args = (char*)malloc(256); // 256 bytes should be enough for two space separated numbers
  assert(args != NULL);
  int argsStrlen = sprintf(args, "%d %d", -1, txPower);
  args[argsStrlen] = '\0';
  return __rscall_ie_set_x(iePath, "/driver/measure", args);
}

int rscall_ie_drv_measure_dual(const char* iePath, const int txPower) {
  // To trigger measurement we write to driver/measure
  char* args = (char*)malloc(256); // 256 bytes should be enough for two space separated numbers
  assert(args != NULL);
  int argsStrlen = sprintf(args, "%d %d", RSCALL_IE_DRV_DUAL_MEASURE_TIMEOUT, txPower);
  args[argsStrlen] = '\0';
  return __rscall_ie_set_x(iePath, "/driver/measure", args);
}


int rscall_ie_get_rssi(const char* iePath, int* pRssi) {
  // Read from `rssi`
  const size_t iePathStrlen = strlen(iePath);
  const char* rssi_path_epilog = "/rssi";
  const size_t rssi_path_epilog_strlen = strlen(rssi_path_epilog);
  char* rssi_path = (char*)malloc(iePathStrlen + rssi_path_epilog_strlen + 1);
  assert(rssi_path != NULL);
  strcpy(rssi_path, iePath);
  strcpy(rssi_path + iePathStrlen, rssi_path_epilog);
  rssi_path[iePathStrlen + rssi_path_epilog_strlen] = '\0';

  FILE* rssi_file = fopen(rssi_path, "r");
  if (rssi_file == NULL) {
    free((void*)rssi_path);
    return -1;
  }
  char rssi[129]; // We should not exceed 128 characters in rssi as all we read is a natural number
  if (fgets(rssi, sizeof(rssi), rssi_file) == NULL) {
    fclose(rssi_file);
    free((void*)rssi_path);
    return -2;
  }
  size_t rssiStrlen = strlen(rssi);
  assert(rssiStrlen > 0 && rssiStrlen < 128); // If we have 128 characters, there is something going on wrong definitely
  fclose(rssi_file);

  //int rssiValue = atoi(rssi);
  //rssi has 2 hex digits, so we need to convert it to decimal
  assert(rssiStrlen == 2);
  int rssiValue = (int)strtol(rssi, NULL, 16);
  *pRssi = rssiValue;
  free((void*)rssi_path);
  return 0;
}

/**
 * TODO Refactor repeating code with rscall_ie_get_rssi and rscall_ie_dir_create
 */
int rscall_ie_get_read_rate(const char* iePath, int* pReadRate) {
  // Read from `rssi`
  const size_t iePathStrlen = strlen(iePath);
  const char* read_rate_path_epilog = "/read_rate";
  const size_t read_rate_path_epilog_strlen = strlen(read_rate_path_epilog);
  char* read_rate_path = (char*)malloc(iePathStrlen + read_rate_path_epilog_strlen + 1);
  assert(read_rate_path != NULL);
  strcpy(read_rate_path, iePath);
  strcpy(read_rate_path + iePathStrlen, read_rate_path_epilog);
  read_rate_path[iePathStrlen + read_rate_path_epilog_strlen] = '\0';

  FILE* read_rate_file = fopen(read_rate_path, "r");
  if (read_rate_file == NULL) {
    free((void*)read_rate_path);
    return -1;
  }
  char read_rate[129]; // We should not exceed 128 characters in rssi as all we read is a natural number
  if (fgets(read_rate, sizeof(read_rate), read_rate_file) == NULL) {
    fclose(read_rate_file);
    free((void*)read_rate_path);
    return -2;
  }
  size_t readRateStrlen = strlen(read_rate);
  assert(readRateStrlen > 0 && readRateStrlen < 128); // If we have 128 characters, there is something going on wrong definitely
  fclose(read_rate_file);

  int readRateValue = atoi(read_rate);
  *pReadRate = readRateValue;
  free((void*)read_rate_path);
  return 0;
}

const char* rscall_ie_get_path(const int index) {
  char* relpath = (char*)malloc(129);
  assert(relpath != NULL);
  int n = snprintf(relpath, 128, "uhf%d", index);
  assert(n > 0 && n < 128);
  assert(relpath[n] == '\0');
  return __rscall_abspath(relpath);
}