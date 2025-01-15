#include "acall.h"
#include "config.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#ifndef ACALL_ANTC_MOUNT_PATH
  #error "ACALL_ANTC_MOUNT_PATH is not defined!"
#endif

/**
 * @warning The returned path needs to be freed with free()
 */
static char* __acall_abspath(const char* __relpath) {
  const char* mountPath = ACALL_ANTC_MOUNT_PATH;
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

// @warning `xPathEpilog` should start with '/'
static int __acall_ant_set_x(const char* path, const char* xPathEpilog, const char* xValue) {
  const size_t pathStrlen = strlen(path);
  assert(path[pathStrlen - 1] != '/');
  const size_t x_path_epilog_strlen = strlen(xPathEpilog);
  if (x_path_epilog_strlen >= 1) {
    assert(xPathEpilog[0] == '/');
  }
  char* x_path = (char*)malloc(pathStrlen + x_path_epilog_strlen + 1);
  assert(x_path != NULL);
  strcpy(x_path, path);
  if (x_path_epilog_strlen >= 1) {
    strcpy(x_path + pathStrlen, xPathEpilog); 
  }
  x_path[pathStrlen + x_path_epilog_strlen] = '\0';

  // Write to x
  FILE* x_file = fopen(x_path, "w");
  if (x_file == NULL) {
    assert(0);
    return -1;
  }
  if (fprintf(x_file, "%s", xValue) <= 0) {
    assert(0 == fclose(x_file));
    assert(0);
    return -2;
  }
  assert(0 == fclose(x_file));
  return 0;
}

int acall_ant_set_enabled(const char* path) {
  return __acall_ant_set_x(path, "", "1");
}

int acall_ant_set_disabled(const char* path) {
  return __acall_ant_set_x(path, "", "0");
}

const char* acall_ant_get_path(const int index) {
  char* relpath = (char*)malloc(129);
  assert(relpath != NULL);
  int n = snprintf(relpath, 128, "ant%d", index);
  assert(n > 0 && n < 128);
  assert(relpath[n] == '\0');
  return __acall_abspath(relpath);
}