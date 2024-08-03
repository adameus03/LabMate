#ifndef LOG_H
#define LOG_H

#define LOG_FATAL    (1)
#define LOG_ERR      (2)
#define LOG_WARN     (3)
#define LOG_INFO     (4)
#define LOG_DBG      (5)
#define LOG_VERBOSE  (6)

#include <stdio.h>
#include <string.h>
#include <time.h>

#define LOG_LEVEL LOG_INFO

static inline void log_timestamp(FILE* stream) {
    char buff[20];
    struct tm *sTm;

    time_t now = time (0);
    sTm = gmtime (&now);

    strftime (buff, sizeof(buff), "%Y-%m-%d %H:%M:%S", sTm);
    fprintf(stream, "%s ", buff);
}

static inline void log_level_str(int level, FILE* stream) {
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

#define LOG(level, stream, is_continuation, should_break_line, ...) do {  \
                            if (level <= LOG_LEVEL) { \
                                if (!is_continuation) { \
                                    log_level_str(level, stream); \
                                    log_timestamp(stream); \
                                    fprintf(stream," (%s:%d): ", __FILE__, __LINE__); \
                                    fprintf(stream, __VA_ARGS__); \
                                } \
                                if (should_break_line) { \
                                    fprintf(stream, "\n"); \
                                } \
                            } \
                        } while (0)

#define LOG_F(...) LOG(LOG_FATAL, stderr, 0, 1, __VA_ARGS__)
#define LOG_E(...) LOG(LOG_ERR, stderr, 0, 1, __VA_ARGS__)
#define LOG_W(...) LOG(LOG_WARN, stderr, 0, 1, __VA_ARGS__)
#define LOG_I(...) LOG(LOG_INFO, stdout, 0, 1, __VA_ARGS__)
#define LOG_D(...) LOG(LOG_DBG, stdout, 0, 1, __VA_ARGS__)
#define LOG_V(...) LOG(LOG_VERBOSE, stdout, 0, 1, __VA_ARGS__)

#define LOG_F_TBC(...) LOG(LOG_FATAL, stderr, 0, 0, __VA_ARGS__)
#define LOG_E_TBC(...) LOG(LOG_ERR, stderr, 0, 0, __VA_ARGS__)
#define LOG_W_TBC(...) LOG(LOG_WARN, stderr, 0, 0, __VA_ARGS__)
#define LOG_I_TBC(...) LOG(LOG_INFO, stdout, 0, 0, __VA_ARGS__)
#define LOG_D_TBC(...) LOG(LOG_DBG, stdout, 0, 0, __VA_ARGS__)
#define LOG_V_TBC(...) LOG(LOG_VERBOSE, stdout, 0, 0, __VA_ARGS__)

#define LOG_F_CTBC(...) LOG(LOG_FATAL, stderr, 1, 0, __VA_ARGS__)
#define LOG_E_CTBC(...) LOG(LOG_ERR, stderr, 1, 0, __VA_ARGS__)
#define LOG_W_CTBC(...) LOG(LOG_WARN, stderr, 1, 0, __VA_ARGS__)
#define LOG_I_CTBC(...) LOG(LOG_INFO, stdout, 1, 0, __VA_ARGS__)
#define LOG_D_CTBC(...) LOG(LOG_DBG, stdout, 1, 0, __VA_ARGS__)
#define LOG_V_CTBC(...) LOG(LOG_VERBOSE, stdout, 1, 0, __VA_ARGS__)

#define LOG_F_CFIN(...) LOG(LOG_FATAL, stderr, 1, 1, __VA_ARGS__)
#define LOG_E_CFIN(...) LOG(LOG_ERR, stderr, 1, 1, __VA_ARGS__)
#define LOG_W_CFIN(...) LOG(LOG_WARN, stderr, 1, 1, __VA_ARGS__)
#define LOG_I_CFIN(...) LOG(LOG_INFO, stdout, 1, 1, __VA_ARGS__)
#define LOG_D_CFIN(...) LOG(LOG_DBG, stdout, 1, 1, __VA_ARGS__)
#define LOG_V_CFIN(...) LOG(LOG_VERBOSE, stdout, 1, 1, __VA_ARGS__)

#endif // LOG_H
