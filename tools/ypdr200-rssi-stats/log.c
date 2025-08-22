#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <plibsys/plibsys.h>
#include "log.h"

PMutex* __log_global_pMutex = NULL;

void log_global_init() {
    if (__log_global_pMutex == NULL) {
        __log_global_pMutex = p_mutex_new();
        if (__log_global_pMutex == NULL) {
            fprintf(stderr, "log_global_init: Failed to create mutex\n");
            exit(EXIT_FAILURE);
        }
        fprintf(stdout, "log_global_init: Log mutex created\n");
    } else {
        fprintf(stderr, "log_global_init: Attempted to call log_global_init() twice\n");
        exit(EXIT_FAILURE);
    }
}

void log_global_deinit() {
    if (__log_global_pMutex != NULL) {
        p_mutex_free(__log_global_pMutex);
        __log_global_pMutex = NULL;
    }
}

void log_begin() {
    assert(NULL != __log_global_pMutex);
    assert(TRUE == p_mutex_lock(__log_global_pMutex));
}

void log_end() {
    assert(NULL != __log_global_pMutex);
    assert(TRUE == p_mutex_unlock(__log_global_pMutex));
}

static void log_timestamp_s(FILE* stream) {
    char buff[20];
    struct tm *sTm;

    time_t now = time (0);
    sTm = gmtime (&now);

    strftime (buff, sizeof(buff), "%Y-%m-%d %H:%M:%S", sTm);
    fprintf(stream, "%s ", buff);
}

static void log_timestamp_precise(FILE* stream) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm* pTm = localtime(&tv.tv_sec);
    assert(pTm != NULL);
    char pTimestamp[30];
    assert(pTm->tm_year < 10000); //We don't want to overflow the buffer. Sorry for the Y10K bug //TODO Fix this before Y10K (:D)
    strftime(pTimestamp, 20, "%Y-%m-%d %H:%M:%S", pTm);
    fprintf(stream, "%s.%06ld ", pTimestamp, tv.tv_usec);
}

void log_timestamp(FILE* stream) {
    log_timestamp_precise(stream);
}

void log_level_str(int level, FILE* stream) {
    switch (level) {
        case LOG_FATAL:
            fprintf(stream, "[FATAL] ");
            break;
        case LOG_ERR:
            fprintf(stream, "[ERROR] ");
            break;
        case LOG_WARN:
            fprintf(stream, "[WARN] ");
            break;
        case LOG_INFO:
            fprintf(stream, "[INFO] ");
            break;
        case LOG_DBG:
            fprintf(stream, "[DEBUG] ");
            break;
        case LOG_VERBOSE:
            fprintf(stream, "[VERBOSE] ");
            break;
        default:
            fprintf(stream, "[UNKNOWN LOG LEVEL] ");
            break;
    }
}


// void log_log(int level, FILE* stream, int is_continuation, int should_break_line, ...) {
//     assert(NULL != __log_global_pMutex);
//     if (level <= LOG_LEVEL) {
//         if (!is_continuation) {
//             assert(TRUE == p_mutex_lock(__log_global_pMutex));
//             fflush(stream);
//             fsync(fileno(stream));
//             log_level_str(level, stream);
//             log_timestamp(stream);
//             fprintf(stream," (%s:%d): ", __FILE__, __LINE__);
//         }
//         va_list args;
//         va_start(args, should_break_line);
//         vfprintf(stream, va_arg(args, const char*), args);
//         va_end(args);
//         if (should_break_line) {
//             fprintf(stream, "\n");
//             fflush(stream);
//             fsync(fileno(stream));
//             assert(TRUE == p_mutex_unlock(__log_global_pMutex));
//         }
//     }
// }


