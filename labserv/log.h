#ifndef LOG_H
#define LOG_H

#define LOG_FATAL    (1)
#define LOG_ERR      (2)
#define LOG_WARN     (3)
#define LOG_INFO     (4)
#define LOG_DBG      (5)
#define LOG_VERBOSE  (6)

#include <stdio.h>
#include <unistd.h>

#define LOG_LEVEL LOG_VERBOSE

//PMutex* __log_global_pMutex = NULL;

/**
 * @warning First initialize plibsys, because this function uses plibsys' PMutex
 */
void log_global_init();

void log_global_deinit();

void log_begin();

void log_end();

void log_timestamp(FILE* stream);

void log_level_str(int level, FILE* stream);

#define LOG(level, stream, is_continuation, should_break_line, ...) do {  \
                            if (level <= LOG_LEVEL) { \
                                if (!is_continuation) { \
                                    log_begin(); \
                                    fflush(stream); \
                                    fsync(fileno(stream)); \
                                    log_level_str(level, stream); \
                                    log_timestamp(stream); \
                                    fprintf(stream," (%s:%d): ", __FILE__, __LINE__); \
                                } \
                                fprintf(stream, __VA_ARGS__); \
                                if (should_break_line) { \
                                    fprintf(stream, "\n"); \
                                    fflush(stream); \
                                    fsync(fileno(stream)); \
                                    log_end(); \
                                } \
                            } \
                        } while (0)

//void log_log(int level, FILE* stream, int is_continuation, int should_break_line, ...);

//#define LOG(level, stream, is_continuation, should_break_line, ...) log_log(level, stream, is_continuation, should_break_line, __VA_ARGS__)

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
