#ifndef LABSERV_CONFIG_H
#define LABSERV_CONFIG_H

#define LABSERV_USE_HTTPS 0
#define LABSERV_USE_MEMCACHED 0
#define LABSERV_IPV4_ADDR "0.0.0.0"
#define LABSERV_IP_PORT 7890
#define LABSERV_H2O_ACCESS_LOG_FILE_PATH "/dev/stdout"

/* Database configuration */
#define LABSERV_DB_CONNINFO "postgresql://lm_u_a6sd78as7d6f78:8730fehoypd9ugcoa(&#*OuDId7&AT*WP8p9yp&W*DU&Gsd;oij;coduwe;yiouwdhfe@labmate.v2024.pl/labmate"
#define LABSERV_DB_INIT_SCRIPT_PATH "sql/init.sql"

/* Logging configuration */
#include "log.h"
#define LOG_LEVEL LOG_VERBOSE


#endif // LABSERV_CONFIG_H