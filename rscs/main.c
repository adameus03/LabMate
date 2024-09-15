#define FUSE_USE_VERSION 30
#define _FILE_OFFSET_BITS  64

#include <fuse.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <plibsys.h>
#include <stdlib.h>
#include <regex.h>
#include <assert.h>
#include "log.h"
#include "uhfd.h"

typedef struct {
	struct {
		regex_t _uhfd_result_m_sid_fin; // /uhfd/result/$sid/fin
		regex_t _m_uhfx_epc; // /uhfX/epc
		regex_t _m_uhfx_access_passwd; // /uhfX/access_passwd
		regex_t _m_uhfx_kill_passwd; // /uhfX/kill_passwd
		regex_t _m_uhfx_flags; // /uhfX/flags
		regex_t _m_uhfx_rssi; // /uhfX/rssi
		regex_t _m_uhfx_read_rate; // /uhfX/read_rate
		//regex_t _m_uhfx_driver_flags; // /uhfX/driver/flags
		//regex_t _m_uhfx_driver_request; // /uhfX/driver/request
		regex_t _m_uhfx_driver_delete; // /uhfX/driver/delete
		regex_t _m_uhfx_driver_embody; // /uhfX/driver/embody
		regex_t _m_uhfx_driver_measure; // /uhfX/driver/measure
		struct {
			regex_t _m_uhfx; // /uhfX
			regex_t _m_uhfx_driver; // /uhfX/driver
			regex_t _uhfd_result_m_sid; // /uhfd/result/$sid
		} d;
	} rgx;
	uhfd_t uhfd;
} main_globals_t;
main_globals_t __main_globals;

static int do_getattr( const char *path, struct stat *st, struct fuse_file_info *fi )
{
	printf( "[getattr] Called\n" );
	printf( "\tAttributes of %s requested\n", path );
	
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
	
	if ((!strcmp( path, "/"))
	    || (!strcmp( path, "/uhfd"))
		|| (0 == regexec( &__main_globals.rgx.d._m_uhfx, path, 0, NULL, 0))
		|| (0 == regexec( &__main_globals.rgx.d._m_uhfx_driver, path, 0, NULL, 0))
		|| (0 == regexec( &__main_globals.rgx.d._uhfd_result_m_sid, path, 0, NULL, 0)))
	{
		st->st_mode = S_IFDIR | 0755;
		st->st_nlink = 2; // Why "two" hardlinks instead of "one"? The answer is here: http://unix.stackexchange.com/a/101536
	}
	else
	{
		st->st_mode = S_IFREG | 0644;
		st->st_nlink = 1;
		st->st_size = 1024;
	}
	
	return 0;
}

static int do_readdir( const char *path, void *buffer, fuse_fill_dir_t filler, off_t offset, struct fuse_file_info *fi, enum fuse_readdir_flags flags )
{
	printf( "--> Getting The List of Files of %s\n", path );
	filler( buffer, ".", NULL, 0, 0 ); // Current Directory
	filler( buffer, "..", NULL, 0, 0 ); // Parent Directory
	
	if (!strcmp( path, "/" )) // If the user is trying to show the files/directories of the root directory show the following
	{
		filler( buffer, "uhfd", NULL, 0, 0 );
		for (unsigned long i = 0; i < __main_globals.uhfd.num_devs; i++) {
			uhfd_dev_t* pDev = &__main_globals.uhfd.pDevs[i];
			if (pDev->flags & UHFD_DEV_FLAG_DELETED) {
				continue;
			}
			char str_dev[25];
			snprintf(str_dev, sizeof(str_dev), "uhf%lu", pDev->devno);
			filler ( buffer, str_dev, NULL, 0, 0 );
		}
	} else if (!strcmp( path, "/uhfd")) {
		filler( buffer, "sid", NULL, 0, 0 );
		filler( buffer, "mkdev", NULL, 0, 0 );
		filler( buffer, "result", NULL, 0, 0 );
	} else if (0 == regexec( &__main_globals.rgx.d._m_uhfx, path, 0, NULL, 0)) {
		filler( buffer, "epc", NULL, 0, 0 );
		filler( buffer, "access_passwd", NULL, 0, 0 );
		filler( buffer, "kill_passwd", NULL, 0, 0 );
		filler( buffer, "flags", NULL, 0, 0 );
		filler( buffer, "rssi", NULL, 0, 0 );
		filler( buffer, "read_rate", NULL, 0, 0 );
		filler( buffer, "driver", NULL, 0, 0 );
	} else if (0 == regexec( &__main_globals.rgx.d._m_uhfx_driver, path, 0, NULL, 0)) {
		filler( buffer, "delete", NULL, 0, 0 );
		filler( buffer, "embody", NULL, 0, 0 );
		filler( buffer, "measure", NULL, 0, 0 );
	} else if (0 == regexec( &__main_globals.rgx.d._uhfd_result_m_sid, path, 0, NULL, 0)) {
		filler( buffer, "value", NULL, 0, 0 );
		filler( buffer, "fin", NULL, 0, 0 );
	}
	
	return 0;
}

static int do_read( const char *path, char *buffer, size_t size, off_t offset, struct fuse_file_info *fi )
{
	printf( "--> Trying to read %s, %u, %u\n", path, offset, size );
	
	regmatch_t matches[2];

	if (!strcmp ( path, "/uhfd/sid")) {
		LOG_E("Read /uhfd/sid: not implemented");
		return -1;
	} else if (!strcmp ( path, "/uhfd/mkdev")) {
		LOG_E("Read /uhfd/mkdev: can't read from mkdev");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._uhfd_result_m_sid_fin, path, 2, matches, 0 )) {
		LOG_I("sid=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Read /uhfd/result/$sid/fin: can't read from fin");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_epc, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Read /uhfX/epc: not implemented");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_access_passwd, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Read /uhfX/access_passwd: not implemented");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_kill_passwd, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Read /uhfX/kill_passwd: not implemented");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_flags, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Read /uhfX/flags: not implemented");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_rssi, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Read /uhfX/rssi: not implemented");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_read_rate, path, 2, matches, 0)) {
		LOG_E("Read /uhfX/read_rate: not implemented");
		return -1;
	// } else if (0 == regexec( &__main_globals.rgx._m_uhfx_driver_flags, path, 2, matches, 0)) {
	// 	LOG_I("Read /uhfX/driver/flags");
	// 	return 0;
	// } else if (0 == regexec( &__main_globals.rgx._m_uhfx_driver_request, path, 2
	// }
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_driver_delete, path, 2, matches, 0)) {
		LOG_E("Read /uhfX/driver/delete: not implemented");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_driver_embody, path, 2, matches, 0)) {
		LOG_E("Read /uhfX/driver/embody: not implemented");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_driver_measure, path, 2, matches, 0)) {
		LOG_E("Read /uhfX/driver/measure: not implemented");
		return -1;
	} else {
		LOG_E("Read %s: not supported", path);
		return -1;
	}
}

static int do_write( const char *path, const char *buffer, size_t size, off_t offset, struct fuse_file_info *fi )
{
	printf( "--> Trying to write %s, %u, %u\n", path, offset, size );
	
	regmatch_t matches[2];

	if (!strcmp( path, "/uhfd/sid")) {
		LOG_E("Write /uhfd/sid: can't write to sid");
		return -1;
	}
	else if (!strcmp( path, "/uhfd/mkdev")) {
		LOG_E("Write /uhfd/mkdev: not implemented");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._uhfd_result_m_sid_fin, path, 2, matches, 0 )) {
		LOG_I("sid=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Write /uhfd/result/$sid/fin: not implemented");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_epc, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Write /uhfX/epc: not implemented");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_access_passwd, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Write /uhfX/access_passwd: not implemented");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_kill_passwd, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Write /uhfX/kill_passwd: not implemented");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_flags, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Write /uhfX/flags: not implemented");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_rssi, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Write /uhfX/rssi: can't write to rssi");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_read_rate, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Write /uhfX/read_rate: can't write to read_rate");
		return -1;
	// } else if (0 == regexec( &__main_globals.rgx._m_uhfx_driver_flags, path, 2, matches, 0)) {
	// 	LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
	// 	LOG_E("Write /uhfX/driver/flags: not implemented");
	// 	return -1;
	// } else if (0 == regexec( &__main_globals.rgx._m_uhfx_driver_request, path, 2, matches, 0)) {
	// 	LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
	// 	LOG_E("Write /uhfX/driver/request: not implemented");
	// 	return -1;
	// } else {
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_driver_delete, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Write /uhfX/driver/delete: not implemented");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_driver_embody, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Write /uhfX/driver/embody: not implemented");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_driver_measure, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Write /uhfX/driver/measure: not implemented");
		return -1;
	} else {
		LOG_E("Write %s: not supported", path);
		return -1;
	}
	
	return strlen( buffer );
}

static struct fuse_operations operations = {
    .getattr	= do_getattr,
    .readdir	= do_readdir,
    .read		= do_read,
};

void main_init() {
	assert(0 == regcomp( &__main_globals.rgx._uhfd_result_m_sid_fin, "^/uhfd/result/([^/]+)/fin$", REG_EXTENDED ));
	assert(0 == regcomp( &__main_globals.rgx._m_uhfx_epc, "^/uhf([0-9]+)/epc$", REG_EXTENDED ));
	assert(0 == regcomp( &__main_globals.rgx._m_uhfx_access_passwd, "^/uhf([0-9]+)/access_passwd$", REG_EXTENDED ));
	assert(0 == regcomp( &__main_globals.rgx._m_uhfx_kill_passwd, "^/uhf([0-9]+)/kill_passwd$", REG_EXTENDED ));
	assert(0 == regcomp( &__main_globals.rgx._m_uhfx_flags, "^/uhf([0-9]+)/flags$", REG_EXTENDED ));
	assert(0 == regcomp( &__main_globals.rgx._m_uhfx_rssi, "^/uhf([0-9]+)/rssi$", REG_EXTENDED ));
	assert(0 == regcomp( &__main_globals.rgx._m_uhfx_read_rate, "^/uhf([0-9]+)/read_rate$", REG_EXTENDED ));
	//assert(0 == regcomp( &__main_globals._m_uhfx_driver_flags, "^/uhf([0-9]+)/driver/flags$", REG_EXTENDED ));
	//assert(0 == regcomp( &__main_globals._m_uhfx_driver_request, "^/uhf([0-9]+)/driver/request$", REG_EXTENDED ));
	assert(0 == regcomp( &__main_globals.rgx._m_uhfx_driver_delete, "^/uhf([0-9]+)/driver/delete$", REG_EXTENDED ));
	assert(0 == regcomp( &__main_globals.rgx._m_uhfx_driver_embody, "^/uhf([0-9]+)/driver/embody$", REG_EXTENDED ));
	assert(0 == regcomp( &__main_globals.rgx._m_uhfx_driver_measure, "^/uhf([0-9]+)/driver/measure$", REG_EXTENDED ));

	assert(0 == regcomp( &__main_globals.rgx.d._m_uhfx, "^/uhf([0-9]+)$", REG_EXTENDED ));
	assert(0 == regcomp( &__main_globals.rgx.d._m_uhfx_driver, "^/uhf([0-9]+)/driver$", REG_EXTENDED ));
	assert(0 == regcomp( &__main_globals.rgx.d._uhfd_result_m_sid, "^/uhfd/result/([^/]+)$", REG_EXTENDED ));
}

int main( int argc, char *argv[] )
{
	p_libsys_init();
	main_init();
	assert(0 == uhfd_init(&__main_globals.uhfd));
	fuse_main( argc, argv, &operations, NULL );
	assert(0 == uhfd_deinit(&__main_globals.uhfd));
	p_libsys_shutdown();
	return 0;
}