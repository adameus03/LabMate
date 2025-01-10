#define FUSE_USE_VERSION 30
#define _FILE_OFFSET_BITS  64

#include "config_checks.h" // Check config.h settings at compile time
#include <stdio.h>
#include <unistd.h>
#include <plibsys.h>
#include <errno.h>
#include <assert.h>
#include <fuse.h>
#include "log.h"
#include "antennactl.h"

typedef struct {
  antennactl_t* pActl;
} main_globals_t;
main_globals_t __main_globals;

static int do_getattr( const char *path, struct stat *st, struct fuse_file_info *fi ) {
  LOG_V("do_getattr: Attributes of %s requested", path);

  const int file_flag = 1;
  const int dir_flag = 2;
  uint8_t ck_flags = 0;

  if (!strcmp(path, "/")) {
    ck_flags |= dir_flag;
  }
  if (!(ck_flags & dir_flag)) {
#if ANTENNACTL_HW_ARCH == ANTENNACTL_HW_ARCH_HMC349_DUAL
    if ((!strcmp(path, "/ant0")) || (!strcmp(path, "/ant1"))) {
      ck_flags |= file_flag;
    }
#endif
  }

  assert(ck_flags != (dir_flag | file_flag));

	if (ck_flags == 0) {
		return -ENOENT;
	} else {
		// GNU's definitions of the attributes (http://www.gnu.org/software/libc/manual/html_node/Attribute-Meanings.html):
		// 		st_uid: 	The user ID of the file’s owner.
		//		st_gid: 	The group ID of the file.
		//		st_atime: 	This is the last access time for the file.
		//		st_mtime: 	This is the time of the last modification to the contents of the file.
		//		st_mode: 	Specifies the mode of the file. This includes file type information (see Testing File Type) and the file permission bits (see Permission Bits).
		//		st_nlink: 	The number of hard links to the file. This count keeps track of how many directories have entries for this file. If the count is ever decremented to zero, then the file itself is discarded as soon 
		//						as no process still holds it open. Symbolic links are not counted in the total.
		//		st_size:	This specifies the size of a regular file in bytes. For files that are really devices this field isn’t usually meaningful. For symbolic links this specifies the length of the file name the link refers to.
		
		st->st_uid = getuid(); // The owner of the file/directory is the user who mounted the filesystem
		st->st_gid = getgid(); // The group of the file/directory is the same as the group of the user who mounted the filesystem
		st->st_atime = time( NULL ); // The last "a"ccess of the file/directory is right now
		st->st_mtime = time( NULL ); // The last "m"odification of the file/directory is right now
		
		if (ck_flags & dir_flag) {
			st->st_mode = S_IFDIR | 0755;
			st->st_nlink = 2; // Why "two" hardlinks instead of "one"? The answer is here: http://unix.stackexchange.com/a/101536
			return 0;
		} else if (ck_flags & file_flag) {
			st->st_mode = S_IFREG | 0644;
			st->st_nlink = 1;
			st->st_size = 1024;
			return 0;
		}
	}
}

static int do_readdir( const char *path, void *buffer, fuse_fill_dir_t filler, off_t offset, struct fuse_file_info *fi, enum fuse_readdir_flags flags ) {
  LOG_V("do_readdir: Called for %s", path);
  filler(buffer, ".", NULL, 0, 0);
  filler(buffer, "..", NULL, 0, 0);

  if (!strcmp(path, "/")) {
#if ANTENNACTL_HW_ARCH == ANTENNACTL_HW_ARCH_HMC349_DUAL
    filler(buffer, "ant0", NULL, 0, 0);
    filler(buffer, "ant1", NULL, 0, 0);
#endif
  }

  return 0;
}

static int do_open( const char* path, struct fuse_file_info* fi ) {
  LOG_V("do_open: Called for %s", path);
  fi->keep_cache = 0;
#if ANTENNACTL_HW_ARCH == ANTENNACTL_HW_ARCH_HMC349_DUAL
  if (!strcmp(path, "/ant0")) {
    fi->fh = ANTENNACTL_TARGET_T0;
    return 0;
  } else if (!strcmp(path, "/ant1")) {
    fi->fh = ANTENNACTL_TARGET_T1;
    return 0;
  }
#endif
  if (strcmp(path, "/")) {
    return -EBADF;
  }
  return 0;
}

static int main_antx_read(antennactl_target_t ant, char* buffer, size_t size, off_t offset, struct fuse_file_info* fi) {
  if (offset != 0) {
    LOG_E("main_antx_read: Invalid offset %u, expected 0", offset);
    return -ESPIPE;
  }
  if (size < 1) {
    LOG_W("main_antx_read: size=%u, returning 0", size);
    return 0;
  }
  assert(fi->fh == ant);

  antennactl_target_t targetAnt;
  antennactl_get_target(__main_globals.pActl, &targetAnt);
  if (targetAnt == ant) {
    buffer[0] = '1';
  } else {
    buffer[0] = '0';
  }
  return 1;
}

static int do_read( const char *path, char *buffer, size_t size, off_t offset, struct fuse_file_info *fi ) {
  LOG_V("do_read: Called for %s, %u, %u", path, offset, size);
#if ANTENNACTL_HW_ARCH == ANTENNACTL_HW_ARCH_HMC349_DUAL
  if (!strcmp(path, "/ant0")) {
    return main_antx_read(ANTENNACTL_TARGET_T0, buffer, size, offset, fi);
  } else if (!strcmp(path, "/ant1")) {
    return main_antx_read(ANTENNACTL_TARGET_T1, buffer, size, offset, fi);
  } 
#endif
  else {
    return -EBADF;
  }
  return 0;
}

static int main_antx_write(antennactl_target_t ant, const char *buffer, size_t size, off_t offset, struct fuse_file_info *fi) {
  if (offset != 0) {
    LOG_E("main_antx_write: Invalid offset %u, expected 0", offset);
    return -ESPIPE;
  }
  if (size == 0) {
    LOG_W("main_antx_write: size=%u, returning 0", size);
    return 0;
  }
  if (size != 1) {
    LOG_E("main_antx_write: Invalid size %u, expected 1", size);
    return -EINVAL;
  }
  assert(fi->fh == ant);

  if (buffer[0] == '1') {
    antennactl_set_target(__main_globals.pActl, ant);
    LOG_D("main_antx_write: Done setting antennactl target as T%d", ant);
    return 1;
  } else if (buffer[0] == '0') {
    LOG_W("main_antx_write: Setting antennactl target as default (T%d)", ANTENNACTL_TARGET_TDEFAULT);
    antennactl_set_target(__main_globals.pActl, ANTENNACTL_TARGET_TDEFAULT);
    LOG_D("main_antx_write: Done setting antennactl target as default (T%d)", ANTENNACTL_TARGET_TDEFAULT);
    return 1;
  } else {
    LOG_E("main_antx_write: Invalid value %c, expected '0' or '1'", buffer[0]);
    return -EINVAL;
  }
}

static int do_write( const char *path, const char *buffer, size_t size, off_t offset, struct fuse_file_info *fi ) {
  LOG_V("do_write: Called for %s, %u, %u", path, offset, size);
#if ANTENNACTL_HW_ARCH == ANTENNACTL_HW_ARCH_HMC349_DUAL
  if (!strcmp(path, "/ant0")) {
    return main_antx_write(ANTENNACTL_TARGET_T0, buffer, size, offset, fi);
  } else if (!strcmp(path, "/ant1")) {
    return main_antx_write(ANTENNACTL_TARGET_T1, buffer, size, offset, fi);
  }
#endif
  else {
    return -EBADF;
  }
  return 0;
}

static struct fuse_operations operations = {
  .getattr	= do_getattr,
  .readdir	= do_readdir,
  .open     = do_open,
  .read		  = do_read,
	.write    = do_write,
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