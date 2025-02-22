/*
    bash/client software --- database
         |
    (filesystem interface)
         |
       main (FUSE) <<< we are here
         |
       uhfd (UHF RFID tag devs driver)
        |
      uhfman (abstracted management for interrogator device / HAL)
        |
     ypdr200 (specific interrogator device driver)
*/

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

#define MAIN_SIDVAL_RINGBUF_SZ 100

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
		PMutex* pSidval_ringbuf_mtx; // Mutex for sidval_ringbuf
	} l; // Locks
	uhfd_t uhfd;
	// unsigned long sid_container_size;
	// unsigned long sid_container_size_flex_max;
	// unsigned long* pSid_container;
	unsigned long sess_counter; // Please use pSess_counter_mtx when accessing. Counter enabling simple sid generation when reading /uhfd/sid and writing /uhfd/mkdev
	struct {
		unsigned long sid;
		unsigned long value;
	} sidval_ringbuf[MAIN_SIDVAL_RINGBUF_SZ];
	unsigned long sidval_ringbuf_head;
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
	return 0;
}

static int main_ulong_from_mkdev_buffer(const char* buffer, size_t size, unsigned long* pUlong_out) {
	char* _buffer = (char*)malloc(size + 1);
	assert(NULL != _buffer);
	memcpy(_buffer, buffer, size);
	_buffer[size] = '\0';
	char* __endp = NULL;
	unsigned long u = strtoul(_buffer, &__endp, 10);
	assert(__endp != NULL);
	if (__endp <= _buffer) { // Don't allow empty strings or strings like "w"
		free(_buffer);
		return -1;
	}
	if ((_buffer[0] == '-') || (_buffer[0] == '+')) { // Don't allow strings like +234 or -19
		free(_buffer);
		return -1;
	}
	if (__endp > (_buffer + 1)) { // Don't allow strings like 0123
		if (_buffer[0] == '0') { 
			free(_buffer);
			return -1;
		}
	}
	if (__endp < (_buffer + size)) { // Don't allow strings like 123w
		free(_buffer);
		return -1;
	}
	free(_buffer);
	*pUlong_out = u;
	return 0;
}

static void main_sidval_ringbuf_init() {
	__main_globals.sidval_ringbuf_head = 0;
	for (unsigned long i = 0; i < MAIN_SIDVAL_RINGBUF_SZ; i++) {
		__main_globals.sidval_ringbuf[i].sid = (unsigned long)-1;
		__main_globals.sidval_ringbuf[i].value = (unsigned long)-1;
	}
}

static void main_sidval_ringbuf_push(unsigned long sid, unsigned long value) {
	assert(TRUE == p_mutex_lock(__main_globals.l.pSidval_ringbuf_mtx));
	__main_globals.sidval_ringbuf[__main_globals.sidval_ringbuf_head].sid = sid;
	__main_globals.sidval_ringbuf[__main_globals.sidval_ringbuf_head].value = value;
	__main_globals.sidval_ringbuf_head = (__main_globals.sidval_ringbuf_head + 1) % MAIN_SIDVAL_RINGBUF_SZ;
	assert(TRUE == p_mutex_unlock(__main_globals.l.pSidval_ringbuf_mtx));
}

// Linear search starting from the head backwards excluding the head (it'll probably usually find the sid quicker than if we used bsearch ( TODO ))
static int main_sidval_ringbuf_find(unsigned long sid, unsigned long* pValue_out) {
	assert(TRUE == p_mutex_lock(__main_globals.l.pSidval_ringbuf_mtx));
	if (__main_globals.sidval_ringbuf_head > 0) {
		for (unsigned long i = __main_globals.sidval_ringbuf_head - 1; i != __main_globals.sidval_ringbuf_head; i--) { // TODO split into two loops for minor optimization?
			if (i == (unsigned long)-1) {
				i = MAIN_SIDVAL_RINGBUF_SZ - 1;
			}
			if (__main_globals.sidval_ringbuf[i].sid == sid) {
				*pValue_out = __main_globals.sidval_ringbuf[i].value;
				assert(TRUE == p_mutex_unlock(__main_globals.l.pSidval_ringbuf_mtx));
				return 0;
			}
		}		
	}
	assert(TRUE == p_mutex_unlock(__main_globals.l.pSidval_ringbuf_mtx));
	return -1;
}

static int main_check_u8_hex_byte(const char bh0, const char bh1) {
	if (!((bh0 >= '0' && bh0 <= '9') || (bh0 >= 'a' && bh0 <= 'f') || (bh0 >= 'A' && bh0 <= 'F'))) {
		return -1;
	}
	if (!((bh1 >= '0' && bh1 <= '9') || (bh1 >= 'a' && bh1 <= 'f') || (bh1 >= 'A' && bh1 <= 'F'))) {
		return -1;
	}
	return 0;
}

static int main_u8buf_from_hex(const char* hex, size_t hex_len, uint8_t* buf, size_t buf_len) {
	if (hex_len % 2 != 0) {
		return -1;
	}
	if (hex_len / 2 != buf_len) {
		return -1;
	}
	for (size_t i = 0; i < hex_len; i += 2) {
		if (0 != main_check_u8_hex_byte(hex[i], hex[i + 1])) { //prevent +, -, etc
			return -1;
		}
		char hex_byte[3];
		hex_byte[0] = hex[i];
		hex_byte[1] = hex[i + 1];
		hex_byte[2] = '\0';
		char* endptr = NULL;
		unsigned long byte = strtoul(hex_byte, &endptr, 16);
		if (endptr != hex_byte + 2) {
			return -1;
		}
		buf[i / 2] = (uint8_t)byte;
	}
	return 0;
}

static int main_hex_from_u8buf(const uint8_t* buf, size_t buf_len, char* hex, size_t hex_len) {
	if (hex_len < (buf_len * 2 + 1)) {
		return -1;
	}
	for (size_t i = 0; i < buf_len; i++) {
		snprintf(hex + i * 2, 3, "%02x", buf[i]);
	}
	return 0;
}

static int do_getattr( const char *path, struct stat *st, struct fuse_file_info *fi )
{
	LOG_V( "[getattr] Called" );
	LOG_V( "\tAttributes of %s requested", path );
	
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
			//unsigned long num_devs = (unsigned long)p_atomic_pointer_get(&__main_globals.uhfd.num_devs);
			//unsigned long num_devs = 0;
			//assert(0 == uhfd_get_num_devs(&__main_globals.uhfd, &num_devs));
			unsigned long devno = (unsigned long)-1;
			if (-1 == main_ulong_from_path_regmatch(path, &matches[1], &devno)) {
				LOG_D("getattr: devno is invalid");
				return -ENOENT;
			}
			assert(devno != (unsigned long)-1);
			//if ((devno >= num_devs) || (devno < 0)) {
			//	LOG_D("getattr: devno=%lu out of range", devno);
			//	return -ENOENT;
			//} else {
			uhfd_dev_t dev;
			int rv = uhfd_get_dev(&__main_globals.uhfd, devno, &dev);
			if (rv != 0) {
				LOG_D("getattr: devno=%lu out of range", devno);
				return -ENOENT;
			}
			uint8_t flags = dev.flags;
			//assert(0 == uhfd_atomic_get_dev_flags(&__main_globals.uhfd, devno, &flags));
			if (flags & UHFD_DEV_FLAG_DELETED) {
				LOG_D("getattr: devno=%lu is deleted", devno);
				return -ENOENT;
			} else {
				ck_flags |= dir_flag;
			}
			//}
		} else if (0 == regexec( &__main_globals.rgx.d._uhfd_result_m_sid, path, 2, matches, 0)) {
			assert(TRUE == p_mutex_lock(__main_globals.l.pSess_counter_mtx));
			//unsigned long sess_counter = (unsigned long)p_atomic_pointer_get(&__main_globals.sess_counter);
			unsigned long sess_counter = __main_globals.sess_counter;
			assert(TRUE == p_mutex_unlock(__main_globals.l.pSess_counter_mtx));
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
				assert(TRUE == p_mutex_lock(__main_globals.l.pSess_counter_mtx));
				//unsigned long sess_counter = (unsigned long)p_atomic_pointer_get(&__main_globals.sess_counter);
				unsigned long sess_counter = __main_globals.sess_counter;
				assert(TRUE == p_mutex_unlock(__main_globals.l.pSess_counter_mtx));
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
				//unsigned long num_devs = (unsigned long)p_atomic_pointer_get(&__main_globals.uhfd.num_devs);
				//unsigned long num_devs = 0;
				//assert(0 == uhfd_get_num_devs(&__main_globals.uhfd, &num_devs));
				unsigned long devno = (unsigned long)-1;
				if (-1 == main_ulong_from_path_regmatch(path, &matches[1], &devno)) {
					LOG_D("getattr: devno is invalid");
					return -ENOENT;
				}
				assert(devno != (unsigned long)-1);
				// if ((devno >= num_devs) || (devno < 0)) {
				// 	LOG_D("getattr: devno=%lu out of range", devno);
				// 	return -ENOENT;
				// } else {
				uhfd_dev_t dev;
				int rv = uhfd_get_dev(&__main_globals.uhfd, devno, &dev);
				if (rv != 0) {
					LOG_D("getattr: devno=%lu out of range", devno);
					return -ENOENT;
				}
				uint8_t flags = dev.flags;
				if (flags & UHFD_DEV_FLAG_DELETED) {
					LOG_D("getattr: devno=%lu is deleted", devno);
					return -ENOENT;
				} else {
					ck_flags |= file_flag;
				}
				// }
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
		//unsigned long num_devs = (unsigned long)p_atomic_pointer_get(&__main_globals.uhfd.num_devs);
		unsigned long num_devs = 0;
		assert(0 == uhfd_get_num_devs(&__main_globals.uhfd, &num_devs)); //[WARNING] Combine into a single critical section with the below loop? This should be not harmful as long as devices are not actually deleted...
		for (unsigned long i = 0; i < num_devs; i++) {
			uhfd_dev_t dev;
			assert(0 == uhfd_get_dev(&__main_globals.uhfd, i, &dev));
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
	} else if (!strcmp( path, "/uhfd/result")) { // TODO remove sids which are no longer in any of the sidval_ringbuf's entries
		assert(TRUE == p_mutex_lock(__main_globals.l.pSess_counter_mtx));
		//unsigned long sess_counter = (unsigned long)p_atomic_pointer_get(&__main_globals.sess_counter);
		unsigned long sess_counter = __main_globals.sess_counter;
		assert(TRUE == p_mutex_unlock(__main_globals.l.pSess_counter_mtx));
		for (unsigned long i = 0; i < sess_counter; i++) {
			char str_sid[30];
			int n = snprintf(str_sid, sizeof(str_sid), "%lu", i);
			assert(n > 0 && n < sizeof(str_sid));
			filler( buffer, str_sid, NULL, 0, 0 );
		}
	} else if (0 == regexec( &__main_globals.rgx.d._m_uhfx, path, 2, matches, 0)) {
		//unsigned long num_devs = (unsigned long)p_atomic_pointer_get(&__main_globals.uhfd.num_devs);
		//unsigned long num_devs = 0;
		//assert(0 == uhfd_get_num_devs(&__main_globals.uhfd, &num_devs));
		unsigned long devno = (unsigned long)-1;
		if (-1 == main_ulong_from_path_regmatch(path, &matches[1], &devno)) {
			LOG_D("getattr: devno is invalid");
			return -ENOENT;
		}
		assert(devno != (unsigned long)-1);
		// if (devno >= num_devs) {
		// 	LOG_D("readdir: devno=%lu out of range", devno);
		// 	return -ENOENT;
		// } else {
		uhfd_dev_t dev;
		int rv = uhfd_get_dev(&__main_globals.uhfd, devno, &dev);
		if (rv != 0) {
			LOG_D("readdir: devno=%lu out of range", devno);
			return -ENOENT;
		}
		uint8_t flags = dev.flags;
		if (flags & UHFD_DEV_FLAG_DELETED) {
			LOG_D("readdir: devno=%lu is deleted", devno);
			return -ENOENT;
		}
		// }
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
		//unsigned long num_devs = (unsigned long)p_atomic_pointer_get(&__main_globals.uhfd.num_devs);
		//unsigned long num_devs = 0;
		//assert(0 == uhfd_get_num_devs(&__main_globals.uhfd, &num_devs));
		//uint8_t dev_flags = (unsigned long)p_atomic_pointer_get(&__main_globals.uhfd.pDevs[devno].flags);
		uhfd_dev_t dev;
		int rv = uhfd_get_dev(&__main_globals.uhfd, devno, &dev);
		if (rv != 0) {
			LOG_D("readdir: devno=%lu out of range", devno);
			return -ENOENT;
		}
		uint8_t dev_flags = dev.flags;
		// if ((devno >= num_devs) || (devno < 0)) {
		// 	LOG_D("readdir: devno=%lu out of range", devno);
		// 	return -ENOENT;
		// } else if (dev_flags & UHFD_DEV_FLAG_DELETED) {
		if (dev_flags & UHFD_DEV_FLAG_DELETED) {
			LOG_D("readdir: devno=%lu is deleted", devno);
			return -ENOENT;
		}
		filler( buffer, "delete", NULL, 0, 0 );
		filler( buffer, "embody", NULL, 0, 0 );
		filler( buffer, "measure", NULL, 0, 0 );
	} else if (0 == regexec( &__main_globals.rgx.d._uhfd_result_m_sid, path, 2, matches, 0)) {
		assert(TRUE == p_mutex_lock(__main_globals.l.pSess_counter_mtx));
		//unsigned long sess_counter = (unsigned long)p_atomic_pointer_get(&__main_globals.sess_counter);
		unsigned long sess_counter = __main_globals.sess_counter;
		assert(TRUE == p_mutex_unlock(__main_globals.l.pSess_counter_mtx));
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
	fi->keep_cache = 0;
	
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
	} else {
		regmatch_t matches[2];
		if (0 == regexec( &__main_globals.rgx._uhfd_result_m_sid_value, path, 2, matches, 0 )) { // TODO handle release?
			assert(TRUE == p_mutex_lock(__main_globals.l.pSess_counter_mtx));
			//unsigned long sess_counter = (unsigned long)p_atomic_pointer_get(&__main_globals.sess_counter);
			unsigned long sess_counter = __main_globals.sess_counter;
			assert(TRUE == p_mutex_unlock(__main_globals.l.pSess_counter_mtx));
			//if (matches)
			unsigned long sid = (unsigned long)-1;
			if (-1 == main_ulong_from_path_regmatch(path, &matches[1], &sid)) {
				LOG_D("open: sid is invalid");
				return -ENOENT;
			}
			assert(sid != (unsigned long)-1);
			if ((sid >= sess_counter) || (sid < 0)) {
				LOG_D("open: sid=%lu out of range", sid);
				return -ENOENT;
			} else {
				if (( fi->flags & 3 ) != O_RDONLY)
				{
					return -EACCES;
				}
				unsigned long val = (unsigned long)-1;
				int rv = main_sidval_ringbuf_find(sid, &val);
				if (rv != 0) {
					LOG_D("open: sid=%lu not found in sidval_ringbuf", sid);
					return -EBUSY;
				}
				assert(val != (unsigned long)-1);
				fi->fh = val; // Store val in the file handle to easily share it accross partial reads
			}
		// Exposing dev copy ptr via fi->fh seams to be the easiest way to share it accross partial reads, thus the common code for files in uhfX
		} else if ((0 == regexec( &__main_globals.rgx._m_uhfx_epc, path, 2, matches, 0))
			|| (0 == regexec( &__main_globals.rgx._m_uhfx_access_passwd, path, 2, matches, 0))
			|| (0 == regexec( &__main_globals.rgx._m_uhfx_kill_passwd, path, 2, matches, 0))
			|| (0 == regexec( &__main_globals.rgx._m_uhfx_flags, path, 2, matches, 0))
			|| (0 == regexec( &__main_globals.rgx._m_uhfx_rssi, path, 2, matches, 0))
			|| (0 == regexec( &__main_globals.rgx._m_uhfx_read_rate, path, 2, matches, 0))
		) {
			//unsigned long num_devs = (unsigned long)p_atomic_pointer_get(&__main_globals.uhfd.num_devs);
			//unsigned long num_devs = 0;
			//assert(0 == uhfd_get_num_devs(&__main_globals.uhfd, &num_devs));
			unsigned long devno = (unsigned long)-1;
			if (-1 == main_ulong_from_path_regmatch(path, &matches[1], &devno)) {
				LOG_D("open: devno is invalid");
				return -ENOENT;
			}
			assert(devno != (unsigned long)-1);
			uhfd_dev_t* pDevCopy = (uhfd_dev_t*)malloc(sizeof(uhfd_dev_t));
			int rv = uhfd_get_dev(&__main_globals.uhfd, devno, pDevCopy);
			if (rv != 0) {
				LOG_D("open: devno=%lu out of range", devno);
				return -ENOENT;
			}
			uint8_t flags = pDevCopy->flags;
			if (flags & UHFD_DEV_FLAG_DELETED) {
				LOG_D("open: devno=%lu is deleted", devno);
				return -ENOENT;
			}
			//fi->fh = devno; // Store devno in the file handle to easily share it accross partial reads
			assert(sizeof(fi->fh) == sizeof(uhfd_dev_t*)); // Ensure that the file handle is large enough to store the pointer
			assert(sizeof(unsigned long) == sizeof(uhfd_dev_t*));
			fi->fh = (unsigned long)pDevCopy;
		}
	}
	
	return 0;
}

static int do_release( const char* path, struct fuse_file_info* fi )
{
	printf( "--> Trying to release %s\n", path );
	// When dev copy is stored under fi->fh, it should be freed here

	regmatch_t matches[2];
	if ((0 == regexec( &__main_globals.rgx._m_uhfx_epc, path, 2, matches, 0))
		|| (0 == regexec( &__main_globals.rgx._m_uhfx_access_passwd, path, 2, matches, 0))
		|| (0 == regexec( &__main_globals.rgx._m_uhfx_kill_passwd, path, 2, matches, 0))
		|| (0 == regexec( &__main_globals.rgx._m_uhfx_flags, path, 2, matches, 0))
		|| (0 == regexec( &__main_globals.rgx._m_uhfx_rssi, path, 2, matches, 0))
		|| (0 == regexec( &__main_globals.rgx._m_uhfx_read_rate, path, 2, matches, 0))
	) {
		// if (( fi->flags & 3 ) != O_RDONLY)
		// {
		// 	return -EACCES;
		// }
		//unsigned long num_devs = (unsigned long)p_atomic_pointer_get(&__main_globals.uhfd.num_devs);
		//unsigned long num_devs = 0;
		//assert(0 == uhfd_get_num_devs(&__main_globals.uhfd, &num_devs));
		unsigned long devno = (unsigned long)-1;
		if (-1 == main_ulong_from_path_regmatch(path, &matches[1], &devno)) {
			LOG_D("release: devno is invalid");
			return -ENOENT;
		}
		assert(devno != (unsigned long)-1);
		uhfd_dev_t* pDevCopy = (uhfd_dev_t*)fi->fh;
		assert(pDevCopy != NULL);
		free(pDevCopy);
		fi->fh = 0U;
	}

	return 0;
}

static int main_uhfx_read(uint8_t* pField, size_t fieldSize, const char* path, char* buffer, size_t size, off_t offset) {
	size_t hexbuf_len = fieldSize * 2 + 1;
	char* hexbuf = (char*)malloc(hexbuf_len);
	assert(0 == main_hex_from_u8buf((const uint8_t*)pField, fieldSize, hexbuf, hexbuf_len));
	size_t len = strlen(hexbuf);
	assert(len == hexbuf_len - 1);
	if (offset >= len) {
		free(hexbuf);
		return 0;
	}
	if (offset + size > len) {
		size = len - offset;
	}
	memcpy(buffer, hexbuf + offset, size);
	free(hexbuf);
	return size;
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
		unsigned long val = (unsigned long) fi->fh; // see do_open
		char str_val[30];
		int n = snprintf(str_val, sizeof(str_val), "%lu", val);
		assert(n > 0 && n < sizeof(str_val));
		size_t len = strlen(str_val);
		if (offset >= len) {
			return 0;
		}
		if (offset + size > len) {
			size = len - offset;
		}
		memcpy(buffer, str_val + offset, size);
		return size;
	} else if (0 == regexec( &__main_globals.rgx._uhfd_result_m_sid_fin, path, 2, matches, 0 )) {
		LOG_I("sid=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Read /uhfd/result/$sid/fin: can't read from fin");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_epc, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		uhfd_dev_t* pd = (uhfd_dev_t*)fi->fh; 
		assert(pd != NULL);
		return main_uhfx_read(pd->epc, sizeof(pd->epc), path, buffer, size, offset);
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_access_passwd, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		uhfd_dev_t* pd = (uhfd_dev_t*)fi->fh;
		assert(pd != NULL);
		return main_uhfx_read(pd->access_passwd, sizeof(pd->access_passwd), path, buffer, size, offset);
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_kill_passwd, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		uhfd_dev_t* pd = (uhfd_dev_t*)fi->fh;
		assert(pd != NULL);
		return main_uhfx_read(pd->kill_passwd, sizeof(pd->kill_passwd), path, buffer, size, offset);
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_flags, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		uhfd_dev_t* pd = (uhfd_dev_t*)fi->fh;
		assert(pd != NULL);
		return main_uhfx_read(&pd->flags, sizeof(pd->flags), path, buffer, size, offset);
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_rssi, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		uhfd_dev_t* pd = (uhfd_dev_t*)fi->fh;
		assert(pd != NULL);
		return main_uhfx_read(&pd->measurement.rssi, sizeof(pd->measurement.rssi), path, buffer, size, offset);
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_read_rate, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		uhfd_dev_t* pd = (uhfd_dev_t*)fi->fh;
		assert(pd != NULL);
		return main_uhfx_read(&pd->measurement.read_rate, sizeof(pd->measurement.read_rate), path, buffer, size, offset);
	// } else if (0 == regexec( &__main_globals.rgx._m_uhfx_driver_flags, path, 2, matches, 0)) {
	// 	LOG_I("Read /uhfX/driver/flags");
	// 	return 0;
	// } else if (0 == regexec( &__main_globals.rgx._m_uhfx_driver_request, path, 2
	// }
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_driver_delete, path, 2, matches, 0)) {
		LOG_E("Read /uhfX/driver/delete: can't read from delete");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_driver_embody, path, 2, matches, 0)) {
		LOG_E("Read /uhfX/driver/embody: can't read from embody");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_driver_measure, path, 2, matches, 0)) {
		LOG_E("Read /uhfX/driver/measure: can't read from measure");
		return -1;
	} else {
		LOG_E("Read %s: not supported", path);
		return -1;
	}
}

static int main_uhfx_write(uint8_t* pField, size_t fieldSize, uhfd_dev_t* pd, const char* path, const char* buffer, size_t size, off_t offset) {
	if (offset != 0) { // We don't support partial writes for simplicity
		LOG_E("Write %s: offset=%lu is invalid", path, offset);
		return -ESPIPE;
	}
	if(pd->flags & UHFD_DEV_FLAG_DELETED) {
		LOG_E("Write %s: can't write to deleted device", path);
		assert((pd->flags & UHFD_DEV_FLAG_DELETED) == 0); // We should not even get here if the device is deleted
		return -EPERM;
	}
	if ((pd->flags & UHFD_DEV_FLAG_EMBODIED)
		&&! (pd->flags & UHFD_DEV_FLAG_IGNORED)) {
		LOG_E("Write %s: can't write to embodied device", path); // For now, we don't allow writing to properties of an embodied non-ignored device (for the sake of simplicity)
		return -EPERM;
	}

	if (pd->flags & UHFD_DEV_FLAG_IGNORED) {
		LOG_W("Write %s: writing to ignored device", path);
	}

	int rv = main_u8buf_from_hex(buffer, size, pField, fieldSize);
	if (rv != 0) {
		LOG_E("Write %s: invalid value", path);
		return -EINVAL;
	}
	uhfd_set_dev(&__main_globals.uhfd, pd->devno, *pd); // Update the device in uhfd (Handles locking, etc.)
	return size;
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
		if (offset != 0) { // We don't support partial writes for simplicity
			LOG_E("Write /uhfd/mkdev: offset=%lu is invalid", offset);
			return -ESPIPE;
		}
		unsigned long sid;
		int rv = main_ulong_from_mkdev_buffer(buffer, size, &sid);
		if (rv != 0) {
			LOG_E("Write /uhfd/mkdev: invalid sid");
			return -EINVAL;
		}
		p_mutex_lock(__main_globals.l.pSess_counter_mtx);
		if (sid >= __main_globals.sess_counter) {
			LOG_E("Write /uhfd/mkdev: sid=%lu is invalid (sess_counter=%lu)", sid, __main_globals.sess_counter);
			p_mutex_unlock(__main_globals.l.pSess_counter_mtx);
			return -EINVAL;
		}
		p_mutex_unlock(__main_globals.l.pSess_counter_mtx);
		unsigned long devnum = (unsigned long)-1;
		rv = uhfd_create_dev(&__main_globals.uhfd, &devnum);
		if ((rv != 0) || (devnum == (unsigned long)-1)) {
			LOG_E("Write /uhfd/mkdev: failed to create device");
			return -EAGAIN;
		}
		main_sidval_ringbuf_push(sid, devnum);
		return size;
	} else if (0 == regexec( &__main_globals.rgx._uhfd_result_m_sid_value, path, 2, matches, 0 )) {
		LOG_I("sid=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Write /uhfd/result/$sid/value: can't write to value");
	} else if (0 == regexec( &__main_globals.rgx._uhfd_result_m_sid_fin, path, 2, matches, 0 )) {
		LOG_I("sid=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		LOG_E("Write /uhfd/result/$sid/fin: not implemented");
		return -1;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_epc, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		uhfd_dev_t* pd = (uhfd_dev_t*)fi->fh;
		assert(pd != NULL);
		return main_uhfx_write(pd->epc, sizeof(pd->epc), pd, path, buffer, size, offset);
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_access_passwd, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		uhfd_dev_t* pd = (uhfd_dev_t*)fi->fh;
		assert(pd != NULL);
		return main_uhfx_write(pd->access_passwd, sizeof(pd->access_passwd), pd, path, buffer, size, offset);
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_kill_passwd, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		uhfd_dev_t* pd = (uhfd_dev_t*)fi->fh;
		assert(pd != NULL);
		return main_uhfx_write(pd->kill_passwd, sizeof(pd->kill_passwd), pd, path, buffer, size, offset);
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_flags, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		uhfd_dev_t* pd = (uhfd_dev_t*)fi->fh;
		assert(pd != NULL);
		return main_uhfx_write(&pd->flags, sizeof(pd->flags), pd, path, buffer, size, offset);
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
		unsigned long devno = (unsigned long)-1;
		if (-1 == main_ulong_from_path_regmatch(path, &matches[1], &devno)) {
			LOG_D("write: devno is invalid");
			return -ENOENT;
		}
		assert(devno != (unsigned long)-1);
		uhfd_dev_t* pDevCopy = (uhfd_dev_t*)malloc(sizeof(uhfd_dev_t));
		int rv = uhfd_get_dev(&__main_globals.uhfd, devno, pDevCopy);
		if (rv != 0) {
			LOG_D("write: devno=%lu out of range", devno);
			free(pDevCopy);
			return -ENOENT;
		}
		uint8_t flags = pDevCopy->flags;
		if (flags & UHFD_DEV_FLAG_DELETED) {
			LOG_D("write: devno=%lu is deleted", devno);
			free(pDevCopy);
			return -ENOENT;
		} else if (flags & UHFD_DEV_FLAG_EMBODIED) {
			LOG_D("write: devno=%lu is already embodied", devno); // TODO allow re-embodying in the future?
			free(pDevCopy);
			return -EPERM;
		}
		rv = uhfd_embody_dev(&__main_globals.uhfd, pDevCopy->devno);
		if (rv != 0) {
			LOG_W("write: failed tag embodiment");
			free(pDevCopy);
			return -EAGAIN;
		}
		rv = uhfd_get_dev(&__main_globals.uhfd, devno, pDevCopy);
		assert(rv == 0);
		assert(pDevCopy->flags & UHFD_DEV_FLAG_EMBODIED);
		free(pDevCopy);
		return size;
	} else if (0 == regexec( &__main_globals.rgx._m_uhfx_driver_measure, path, 2, matches, 0)) {
		LOG_I("uhf=%.*s", matches[1].rm_eo - matches[1].rm_so, path + matches[1].rm_so);
		unsigned long devno = (unsigned long)-1;
		if (-1 == main_ulong_from_path_regmatch(path, &matches[1], &devno)) {
			LOG_D("write: devno is invalid");
			return -ENOENT;
		}
		assert(devno != (unsigned long)-1);
		uhfd_dev_t* pDevCopy = (uhfd_dev_t*)malloc(sizeof(uhfd_dev_t));
		int rv = uhfd_get_dev(&__main_globals.uhfd, devno, pDevCopy);
		if (rv != 0) {
			LOG_D("write: devno=%lu out of range", devno);
			free(pDevCopy);
			return -ENOENT;
		}
		uint8_t flags = pDevCopy->flags;
		if (flags & UHFD_DEV_FLAG_DELETED) {
			LOG_D("write: devno=%lu is deleted", devno);
			free(pDevCopy);
			return -ENOENT;
		} else if (! (flags & UHFD_DEV_FLAG_EMBODIED)) {
			LOG_D("write: devno=%lu is not embodied", devno); // TODO print FUSE path in these logs?
			free(pDevCopy);
			return -EPERM;
		}

		assert(offset == 0); // We don't support partial writes for simplicity
		// User writes to this file:
		//	a) a natural number to request a full measurement (rssi + read_rate)
		//  b) -1 to request a quick measurement (rssi only)

		// Read the value from the buffer
		char str_val[30];
		memcpy(str_val, buffer, size);
		str_val[size] = '\0';
		long tm_val; // us timeout for full measurement or -1 for quick measurement
		float tx_p_val = 26.0f; // optional tx power adjustment (new power value in dBm)
		int n = sscanf(str_val, "%ld %f", &tm_val, &tx_p_val);
		if (n < 1) {
			LOG_E("Write %s: invalid value", path);
			free(pDevCopy);
			return -EINVAL;
		} else if (n == 2) {
			LOG_D("Write %s: tx_p_val=%f", path, tx_p_val);
		}
		assert (n == 1 || n == 2);
		if (tm_val < -1) {
			LOG_E("Write %s: invalid value", path);
			free(pDevCopy);
			return -EINVAL;
		}
		assert(pDevCopy->flags & UHFD_DEV_FLAG_EMBODIED);//TODO or remove it if neccessary here and deeper in the code to enable testing for duplicate tag EPCs ?
		if (tm_val == -1) {
			// Request a quick measurement
			rv = uhfd_quick_measure_dev_rssi(&__main_globals.uhfd, pDevCopy->devno, tx_p_val);
			if (rv != 0) {
				LOG_W("write: failed quick measurement");
				free(pDevCopy);
				return -EAGAIN;
			}
		} else {
			// Request a full measurement
			rv = uhfd_measure_dev(&__main_globals.uhfd, pDevCopy->devno, tm_val, tx_p_val);
			if (rv != 0) {
				LOG_W("write: failed measurement");
				free(pDevCopy);
				return -EAGAIN;
			}
		}
		free(pDevCopy);
		return size;
	} else {
		LOG_E("Write %s: not supported", path);
		return -1;
	}
	LOG_W("Write %s: Invalid path", path);
	return -ENOENT;
}

static struct fuse_operations operations = {
    .getattr	= do_getattr,
    .readdir	= do_readdir,
    .read		= do_read,
	.write      = do_write,
	.open       = do_open,
	//.flush 	= NULL,
	.release    = do_release,
	//.destroy  = NULL,
	//.init	    = NULL
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
	assert(NULL != (__main_globals.l.pSidval_ringbuf_mtx = p_mutex_new()));

	main_sidval_ringbuf_init();

	__main_globals.sess_counter = 0;
}

void main_deinit() {
	// Locking deinit
	p_mutex_free(__main_globals.l.pSess_counter_mtx);
	p_mutex_free(__main_globals.l.pSidval_ringbuf_mtx);

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
	//assert(sizeof(unsigned long) == sizeof(uint64_t));
	p_libsys_init();
	main_init();
	assert(0 == uhfd_init(&__main_globals.uhfd));
	fuse_main( argc, argv, &operations, NULL );
	assert(0 == uhfd_deinit(&__main_globals.uhfd));
	main_deinit();
	p_libsys_shutdown();
	return 0;
}