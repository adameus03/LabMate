#define FUSE_USE_VERSION 30
#define _FILE_OFFSET_BITS  64

#include "config_checks.h" // Check config.h settings at compile time
#include <stdio.h>
#include <plibsys.h>
#include <fuse.h>
#include "antennactl.h"

typedef struct {
  antennactl_t* pActl;
} main_globals_t;
main_globals_t __main_globals;

static struct fuse_operations operations = {
  // .getattr	= do_getattr,
  // .readdir	= do_readdir,
  // .read		  = do_read,
	// .write    = do_write,
	// .open     = do_open,
	// .release  = do_release,
};

int main(int argc, char *argv[]) {
  p_libsys_init();
  __main_globals.pActl = antennactl_new();
  antennactl_init(__main_globals.pActl);
  
  fuse_main( argc, argv, &operations, NULL );
  
  antennactl_deinit(__main_globals.pActl);
  antennactl_free(__main_globals.pActl);
  p_libsys_shutdown();
  return 0;
}