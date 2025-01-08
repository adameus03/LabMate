#include "rscall.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "config.h"

#ifndef RSCALL_RSCS_MOUNT_PATH
    #error "RSCALL_RSCS_MOUNT_PATH is not defined!"
#endif

char* __rscall_abspath(const char* __relpath) {
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
    return NULL;
  }
  char sid[129]; // We should not exceed 128 characters in SID as all we read is a natural number
  if (fgets(sid, sizeof(sid), sid_file) == NULL) {
    fclose(sid_file);
    return NULL;
  }
  size_t sidStrlen = strlen(sid);
  assert(sidStrlen > 0 && sidStrlen < 128); // If we have 128 characters, there is something going on wrong definitely
  fclose(sid_file);

  // Write to uhfd/mkdev to trigger the creation of a new device
  FILE* mkdev_file = fopen(mkdev_path, "w");
  if (mkdev_file == NULL) {
    assert(0);
    return NULL;
  }
  if (fprintf(mkdev_file, "%s", sid) <= 0) {
    assert(0 == fclose(mkdev_file));
    assert(0);
    return NULL;
  }
  assert(0 == fclose(mkdev_file));

  // Read from result/$sid/value to get the devno
  char* devno_path_prolog = __rscall_abspath("uhfd/result/");
  size_t devno_path_prolog_strlen = strlen(devno_path_prolog);
  char* devno_path_epilog = "/value";
  char* devno_path = (char*)malloc(devno_path_prolog_strlen + sidStrlen + strlen(devno_path_epilog) + 1);
  strcpy(devno_path, devno_path_prolog);
  strcpy(devno_path + devno_path_prolog_strlen, sid);
  strcpy(devno_path + devno_path_prolog_strlen + sidStrlen, devno_path_epilog);
  devno_path[devno_path_prolog_strlen + sidStrlen] = '\0';
  FILE* devno_file = fopen(devno_path, "r");
  if (devno_file == NULL) {
    assert(0);
    return NULL;
  }
  char devno[129]; // We should not exceed 128 characters in devno as all we read is a natural number
  if (fgets(devno, sizeof(devno), devno_file) == NULL) {
    fclose(devno_file);
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
  strcpy(ie_dir_path, ie_dir_path_prolog);
  strcpy(ie_dir_path + ie_dir_path_prolog_strlen, devno);
  ie_dir_path[ie_dir_path_prolog_strlen + devnoStrlen] = '\0';

  return ie_dir_path;
}

// @warning `xPathEpilog` should start with '/'
static int rscall_ie_set_x(const char* iePath, const char* xPathEpilog, const char* xValue) {
  const size_t iePathStrlen = strlen(iePath);
  assert(iePath[iePathStrlen - 1] != '/');
  const size_t x_path_epilog_strlen = strlen(xPathEpilog);
  assert(xPathEpilog[0] == '/');
  char* access_passwd_path = (char*)malloc(iePathStrlen + x_path_epilog_strlen + 1);
  strcpy(access_passwd_path, iePath);
  strcpy(access_passwd_path + iePathStrlen, xPathEpilog);
  access_passwd_path[iePathStrlen + x_path_epilog_strlen] = '\0';

  // Write to x
  FILE* access_passwd_file = fopen(access_passwd_path, "w");
  if (access_passwd_file == NULL) {
    assert(0);
    return -1;
  }
  if (fprintf(access_passwd_file, "%s", xValue) <= 0) {
    assert(0 == fclose(access_passwd_file));
    assert(0);
    return -1;
  }
  assert(0 == fclose(access_passwd_file));
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
  return rscall_ie_set_x(iePath, "/access_passwd", passwd);
}

int rscall_ie_set_kill_passwd(const char* iePath, const char* passwd) {
  return rscall_ie_set_x(iePath, "/kill_passwd", passwd);
}

int rscall_ie_set_epc(const char* iePath, const char* epc) {
  return rscall_ie_set_x(iePath, "/epc", epc);
}

int rscall_ie_set_flags(const char* iePath, const char* flags) {
  return rscall_ie_set_x(iePath, "/flags", flags);
}

int rscall_ie_drv_embody(const char* iePath) {
  return rscall_ie_set_x(iePath, "/driver/embody", "1");
}

int rscall_ie_drv_measure_quick(const char* iePath, const int txPower) {
  // To trigger measurement we write to driver/measure
  char* args = (char*)malloc(256); // 256 bytes should be enough for two space separated numbers
  int argsStrlen = sprintf(args, "%d %d", -1, txPower);
  args[argsStrlen] = '\0';
  return rscall_ie_set_x(iePath, "/driver/measure", args);
}

int rscall_ie_drv_measure_dual(const char* iePath, const int txPower) {
  // To trigger measurement we write to driver/measure
  char* args = (char*)malloc(256); // 256 bytes should be enough for two space separated numbers
  int argsStrlen = sprintf(args, "%d %d", RSCALL_IE_DRV_DUAL_MEASURE_TIMEOUT, txPower);
  args[argsStrlen] = '\0';
  return rscall_ie_set_x(iePath, "/driver/measure", args);
}
