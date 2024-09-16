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

// #define __SID_CONTAINER_INITIAL_SIZE 16
// #define __SID_CONTAINER_SCALING_FACTOR 1.5

typedef struct {
	struct { // Regular expressions
		regex_t _uhfd_result_m_sid_value; // /uhfd/result/$sid/value
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
	} rgx; // Regular expressions
	struct { // Locks
		PMutex* pSess_counter_mtx; // Mutex for sess_counter
	} l; // Locks
	uhfd_t uhfd;
	// unsigned long sid_container_size;
	// unsigned long sid_container_size_flex_max;
	// unsigned long* pSid_container;
	unsigned long sess_counter; // Counter enabling simple sid generation when reading /uhfd/sid and writing /uhfd/mkdev
} main_globals_t;
main_globals_t __main_globals;

static int main_ulong_from_path_regmatch(const char* path, regmatch_t* pRegmatch, unsigned long* pUlong_out) {
	//unsigned long devno = strtoul(path + matches[1].rm_so, NULL, 10);
	const char* __startp = path + pRegmatch->rm_so;
	char* __endp = NULL;
	unsigned long u = strtoul(path + pRegmatch->rm_so, &__endp, 10);
	assert(__endp != NULL);
	if (__endp <= __startp) { // Don't allow empty strings or strings like "w"
		return -1;
	}
	if ((__startp[0] == '-') || (__startp[0] == '+')) { // Don't allow strings like +234 or -19
		return -1;
	}
	if (__endp > (__startp + 1)) { // Don't allow strings like 0123
		if (__startp[0] == '0') { 
			return -1;
		}
	}
	if ((__endp[0] != '\0') && (__endp[0] != '/')) { // Don't allow strings like 123w, but allow 123/ (as a part of a path)
		return -1;
	}
	*pUlong_out = u;
}

static int do_getattr( const char *path, struct stat *st, struct fuse_file_info *fi )
{
	printf( "[getattr] Called\n" );
	printf( "\tAttributes of %s requested\n", path );
	
	const int file_flag = 1;
	const int dir_flag = 2;
	uint8_t ck_flags = 0;

	if ((!strcmp( path, "/"))
	    || (!strcmp( path, "/uhfd"))
		|| (!strcmp( path, "/uhfd/result"))
		// || (0 == regexec( &__main_globals.rgx.d._m_uhfx, path, 0, NULL, 0))
		// || (0 == regexec( &__main_globals.rgx.d._m_uhfx_driver, path, 0, NULL, 0))
		// || (0 == regexec( &__main_globals.rgx.d._uhfd_result_m_sid, path, 0, NULL, 0))
		) {
		ck_flags |= dir_flag;
	} else {
		regmatch_t matches[2];
		if ((0 == regexec( &__main_globals.rgx.d._m_uhfx, path, 2, matches, 0))
		|| (0 == regexec( &__main_globals.rgx.d._m_uhfx_driver, path, 2, matches, 0))) {
			unsigned long num_devs = (unsigned long)p_atomic_pointer_get(&__main_globals.uhfd.num_devs);
			unsigned long devno = (unsigned long)-1;
			if (-1 == main_ulong_from_path_regmatch(path, &matches[1], &devno)) {
				LOG_D("getattr: devno is invalid");
				return -ENOENT;
			}
			assert(devno != (unsigned long)-1);
			if ((devno >= num_devs) || (devno < 0)) {
				LOG_D("getattr: devno=%lu out of range", devno);
				return -ENOENT;
			} else {
				uint8_t flags;
				assert(0 == uhfd_atomic_get_dev_flags(&__main_globals.uhfd, devno, &flags));
				if (flags & UHFD_DEV_FLAG_DELETED) {
					LOG_D("getattr: devno=%lu is deleted", devno);
					return -ENOENT;
				} else {
					ck_flags |= dir_flag;
				}
			}
		} else if (0 == regexec( &__main_globals.rgx.d._uhfd_result_m_sid, path, 2, matches, 0)) {
			unsigned long sess_counter = (unsigned long)p_atomic_pointer_get(&__main_globals.sess_counter);
			//if (matches)
			unsigned long sid = (unsigned long)-1;
			if (-1 == main_ulong_from_path_regmatch(path, &matches[1], &sid)) {
				LOG_D("getattr: sid is invalid");
				return -ENOENT;
			}
			assert(sid != (unsigned long)-1);
			if ((sid >= sess_counter) || (sid < 0)) {
				LOG_D("getattr: sid=%lu out of range", sid);
				return -ENOENT;
			} else {
				ck_flags |= dir_flag;
			}
		}
	}

	if (!(ck_flags & dir_flag)) {
		if ((!strcmp ( path, "/uhfd/sid"))
		|| (!strcmp ( path, "/uhfd/mkdev"))
		// || (0 == regexec( &__main_globals.rgx._uhfd_result_m_sid_value, path, 0, NULL, 0))
		// || (0 == regexec( &__main_globals.rgx._uhfd_result_m_sid_fin, path, 0, NULL, 0))
		// || (0 == regexec( &__main_globals.rgx._m_uhfx_epc, path, 0, NULL, 0))
		// || (0 == regexec( &__main_globals.rgx._m_uhfx_access_passwd, path, 0, NULL, 0))
		// || (0 == regexec( &__main_globals.rgx._m_uhfx_kill_passwd, path, 0, NULL, 0))
		// || (0 == regexec( &__main_globals.rgx._m_uhfx_flags, path, 0, NULL, 0))
		// || (0 == regexec( &__main_globals.rgx._m_uhfx_rssi, path, 0, NULL, 0))
		// || (0 == regexec( &__main_globals.rgx._m_uhfx_read_rate, path, 0, NULL, 0))
		// // || (0 == regexec( &__main_globals.rgx._m_uhfx_driver_flags, path, 0, NULL, 0))
		// // || (0 == regexec( &__main_globals.rgx._m_uhfx_driver_request, path, 0, NULL, 0))
		// || (0 == regexec( &__main_globals.rgx._m_uhfx_driver_delete, path, 0, NULL, 0))
		// || (0 == regexec( &__main_globals.rgx._m_uhfx_driver_embody, path, 0, NULL, 0))
		// || (0 == regexec( &__main_globals.rgx._m_uhfx_driver_measure, path, 0, NULL, 0))
		) {
			ck_flags |= file_flag;
		} else {
			regmatch_t matches[2];
			if ((0 == regexec( &__main_globals.rgx._uhfd_result_m_sid_value, path, 2, matches, 0))
			|| (0 == regexec( &__main_globals.rgx._uhfd_result_m_sid_fin, path, 2, matches, 0))) {
				unsigned long sess_counter = (unsigned long)p_atomic_pointer_get(&__main_globals.sess_counter);
				unsigned long sid = (unsigned long)-1;
				if (-1 == main_ulong_from_path_regmatch(path, &matches[1], &sid)) {
					LOG_D("getattr: sid is invalid");
					return -ENOENT;
				}
				assert(sid != (unsigned long)-1);
				if ((sid >= sess_counter) || (sid < 0)) {
					LOG_D("getattr: sid=%lu out of range", sid);
					return -ENOENT;
				} else {
					ck_flags |= file_flag;
				}
			} else if ((0 == regexec( &__main_globals.rgx._m_uhfx_epc, path, 2, matches, 0))
				|| (0 == regexec( &__main_globals.rgx._m_uhfx_access_passwd, path, 2, matches, 0))
				|| (0 == regexec( &__main_globals.rgx._m_uhfx_kill_passwd, path, 2, matches, 0))
				|| (0 == regexec( &__main_globals.rgx._m_uhfx_flags, path, 2, matches, 0))
				|| (0 == regexec( &__main_globals.rgx._m_uhfx_rssi, path, 2, matches, 0))
				|| (0 == regexec( &__main_globals.rgx._m_uhfx_read_rate, path, 2, matches, 0))
				// || (0 == regexec( &__main_globals.rgx._m_uhfx_driver_flags, path, 2, matches, 0))
				// || (0 == regexec( &__main_globals.rgx._m_uhfx_driver_request, path, 0, NULL, 0))
				|| (0 == regexec( &__main_globals.rgx._m_uhfx_driver_delete, path, 2, matches, 0))
				|| (0 == regexec( &__main_globals.rgx._m_uhfx_driver_embody, path, 2, matches, 0))
				|| (0 == regexec( &__main_globals.rgx._m_uhfx_driver_measure, path, 2, matches, 0))) {
				unsigned long num_devs = (unsigned long)p_atomic_pointer_get(&__main_globals.uhfd.num_devs);
				unsigned long devno = (unsigned long)-1;
				if (-1 == main_ulong_from_path_regmatch(path, &matches[1], &devno)) {
					LOG_D("getattr: devno is invalid");
					return -ENOENT;
				}
				assert(devno != (unsigned long)-1);
				if ((devno >= num_devs) || (devno < 0)) {
					LOG_D("getattr: devno=%lu out of range", devno);
					return -ENOENT;
				} else {
					uint8_t flags;
					assert(0 == uhfd_atomic_get_dev_flags(&__main_globals.uhfd, devno, &flags));
					if (flags & UHFD_DEV_FLAG_DELETED) {
						LOG_D("getattr: devno=%lu is deleted", devno);
						return -ENOENT;
					} else {
						ck_flags |= file_flag;
					}
				}
			}
		}
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

static int do_readdir( const char *path, void *buffer, fuse_fill_dir_t filler, off_t offset, struct fuse_file_info *fi, enum fuse_readdir_flags flags )
{
	printf( "--> Getting The List of Files of %s\n", path );
	filler( buffer, ".", NULL, 0, 0 ); // Current Directory
	filler( buffer, "..", NULL, 0, 0 ); // Parent Directory
	
	regmatch_t matches[2];

	if (!strcmp( path, "/" )) // If the user is trying to show the files/directories of the root directory show the following
	{
		filler( buffer, "uhfd", NULL, 0, 0 );
		unsigned long num_devs = (unsigned long)p_atomic_pointer_get(&__main_globals.uhfd.num_devs);
		for (unsigned long i = 0; i < num_devs; i++) {
			uhfd_dev_t dev;
			assert(0 == uhfd_atomic_get_dev(&__main_globals.uhfd, i, &dev));
			if (dev.flags & UHFD_DEV_FLAG_DELETED) {
				continue;
			}
			char str_dev[25];
			snprintf(str_dev, sizeof(str_dev), "uhf%lu", dev.devno);
			filler ( buffer, str_dev, NULL, 0, 0 );
		}
	} else if (!strcmp( path, "/uhfd")) {
		filler( buffer, "sid", NULL, 0, 0 );
		filler( buffer, "mkdev", NULL, 0, 0 );
		filler( buffer, "result", NULL, 0, 0 );
	} else if (!strcmp( path, "/uhfd/result")) {
		unsigned long sess_counter = (unsigned long)p_atomic_pointer_get(&__main_globals.sess_counter);
		for (unsigned long i = 0; i < sess_counter; i++) {
			char str_sid[30];
			int n = snprintf(str_sid, sizeof(str_sid), "%lu", i);
			assert(n > 0 && n < sizeof(str_sid));
			filler( buffer, str_sid, NULL, 0, 0 );
		}
	} else if (0 == regexec( &__main_globals.rgx.d._m_uhfx, path, 2, matches, 0)) {
		unsigned long num_devs = (unsigned long)p_atomic_pointer_get(&__main_globals.uhfd.num_devs);
		unsigned long devno = (unsigned long)-1;
		if (-1 == main_ulong_from_path_regmatch(path, &matches[1], &devno)) {
			LOG_D("getattr: devno is invalid");
			return -ENOENT;
		}
		assert(devno != (unsigned long)-1);
		if (devno >= num_devs) {
			LOG_D("readdir: devno=%lu out of range", devno);
			return -ENOENT;
		} else {
			uint8_t flags;
			assert(0 == uhfd_atomic_get_dev_flags(&__main_globals.uhfd, devno, &flags));
			if (flags & UHFD_DEV_FLAG_DELETED) {
				LOG_D("readdir: devno=%lu is deleted", devno);
				return -ENOENT;
			}
		}
		filler( buffer, "epc", NULL, 0, 0 );
		filler( buffer, "access_passwd", NULL, 0, 0 );
		filler( buffer, "kill_passwd", NULL, 0, 0 );
		filler( buffer, "flags", NULL, 0, 0 );
		filler( buffer, "rssi", NULL, 0, 0 );
		filler( buffer, "read_rate", NULL, 0, 0 );
		filler( buffer, "driver", NULL, 0, 0 );
	} else if (0 == regexec( &__main_globals.rgx.d._m_uhfx_driver, path, 2, matches, 0)) {
		unsigned long devno = (unsigned long)-1;
		if (-1 == main_ulong_from_path_regmatch(path, &matches[1], &devno)) {
			LOG_D("getattr: devno is invalid");
			return -ENOENT;
		}
		assert(devno != (unsigned long)-1);
		unsigned long num_devs = (unsigned long)p_atomic_pointer_get(&__main_globals.uhfd.num_devs);
		uint8_t dev_flags = (unsigned long)p_atomic_pointer_get(&__main_globals.uhfd.pDevs[devno].flags);
		if ((devno >= num_devs) || (devno < 0)) {
			LOG_D("readdir: devno=%lu out of range", devno);
			return -ENOENT;
		} else if (dev_flags & UHFD_DEV_FLAG_DELETED) {
			LOG_D("readdir: devno=%lu is deleted", devno);
			return -ENOENT;
		}
		filler( buffer, "delete", NULL, 0, 0 );
		filler( buffer, "embody", NULL, 0, 0 );
		filler( buffer, "measure", NULL, 0, 0 );
	} else if (0 == regexec( &__main_globals.rgx.d._uhfd_result_m_sid, path, 2, matches, 0)) {
		unsigned long sess_counter = (unsigned long)p_atomic_pointer_get(&__main_globals.sess_counter);
		unsigned long sid = (unsigned long)-1;
		if (-1 == main_ulong_from_path_regmatch(path, &matches[1], &sid)) {
			LOG_D("getattr: sid is invalid");
			return -ENOENT;
		}
		assert(sid != (unsigned long)-1);
		if ((sid >= sess_counter) || (sid < 0)) {
			LOG_D("readdir: sid=%lu out of range", sid);
			return -ENOENT;
		}
		filler( buffer, "value", NULL, 0, 0 );
		filler( buffer, "fin", NULL, 0, 0 );
	}
	
	return 0;
}

static int do_open( const char* path, struct fuse_file_info* fi )
{
	printf( "--> Trying to open %s\n", path );
	
	if ( strcmp( path, "/uhfd/sid" ) == 0 )
	{
		if (( fi->flags & 3 ) != O_RDONLY)
		{
			return -EACCES;
		}
		assert(TRUE == p_mutex_lock(__main_globals.l.pSess_counter_mtx));
		fi->fh = __main_globals.sess_counter; // Store the session counter value as session id (sid) in the file handle to easily share it accross partial reads
		__main_globals.sess_counter++;
		assert(TRUE == p_mutex_unlock(__main_globals.l.pSess_counter_mtx));
	}
	
	return 0;
}

static int do_flush( const char* path, struct fuse_file_info* fi )
{
	printf( "--> Trying to flush %s\n", path );
	// Nothing to do for now, may change in the future
	return 0;
}

static int do_read( const char *path, char *buffer, size_t size, off_t offset, struct fuse_file_info *fi )
{
	printf( "--> Trying to read %s, %u, %u\n", path, offset, size );
	
	regmatch_t matches[2];

	if (!strcmp ( path, "/uhfd/sid")) {
		unsigned long sid = (unsigned long) fi->fh; // see do_open
		char str_sid[30];
		int n = snprintf(str_sid, sizeof(str_sid), "%lu", sid);
		assert(n > 0 && n < sizeof(str_sid));
		size_t len = strlen(str_sid);
		if (offset >= len) {
			return 0;
		}
		if (offset + size > len) {
			size = len - offset;
		}
		memcpy(buffer, str_sid + offset, size);
		return size;
	} else if (!strcmp ( path, "/uhfd/mkdev")) {
		LOG_E("Read /uhfd/mkdev: can't read from mkdev");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._uhfd_result_m_sid_value, path, 2, matches, 0 )) {
		LOG_I("sid=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Read /uhfd/result/$sid/value: not implemented");
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
	} else if (0 == regexec( &__main_globals.rgx._uhfd_result_m_sid_value, path, 2, matches, 0 )) {
		LOG_I("sid=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Write /uhfd/result/$sid/value: can't write to value");
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
	.write      = do_write,
	.open       = do_open,
	.flush 	    = do_flush
};

void main_init() {
	// Regex init
	assert(0 == regcomp( &__main_globals.rgx._uhfd_result_m_sid_value, "^/uhfd/result/([^/]+)/value$", REG_EXTENDED ));
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

	// Locking init
	assert(NULL != (__main_globals.l.pSess_counter_mtx = p_mutex_new()));

	__main_globals.sess_counter = 0;
}

void main_deinit() {
	// Locking deinit
	p_mutex_free(__main_globals.l.pSess_counter_mtx);

	// Regex deinit
	regfree( &__main_globals.rgx._uhfd_result_m_sid_value );
	regfree( &__main_globals.rgx._uhfd_result_m_sid_fin );
	regfree( &__main_globals.rgx._m_uhfx_epc );
	regfree( &__main_globals.rgx._m_uhfx_access_passwd );
	regfree( &__main_globals.rgx._m_uhfx_kill_passwd );
	regfree( &__main_globals.rgx._m_uhfx_flags );
	regfree( &__main_globals.rgx._m_uhfx_rssi );
	regfree( &__main_globals.rgx._m_uhfx_read_rate );
	//regfree( &__main_globals._m_uhfx_driver_flags );
	//regfree( &__main_globals._m_uhfx_driver_request );
	regfree( &__main_globals.rgx._m_uhfx_driver_delete );
	regfree( &__main_globals.rgx._m_uhfx_driver_embody );
	regfree( &__main_globals.rgx._m_uhfx_driver_measure );

	regfree( &__main_globals.rgx.d._m_uhfx );
	regfree( &__main_globals.rgx.d._m_uhfx_driver );
	regfree( &__main_globals.rgx.d._uhfd_result_m_sid );
}

int main( int argc, char *argv[] )
{
	p_libsys_init();
	main_init();
	assert(0 == uhfd_init(&__main_globals.uhfd));
	fuse_main( argc, argv, &operations, NULL );
	assert(0 == uhfd_deinit(&__main_globals.uhfd));
	main_deinit();
	p_libsys_shutdown();
	return 0;
}