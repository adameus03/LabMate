/*
  IMPORTANT: This is a sample configuration file. Before building the project, you should copy this file to config.h and modify it to suit your needs.
*/

#ifndef LABSERV_CONFIG_H
#define LABSERV_CONFIG_H

#define LABSERV_USE_HTTPS 0
#define LABSERV_USE_MEMCACHED 0
#define LABSERV_IPV4_ADDR "0.0.0.0"
#define LABSERV_IP_PORT 7890
#define LABSERV_H2O_ACCESS_LOG_FILE_PATH "/dev/stdout"

/* Database configuration */
#define LABSERV_DB_CONNINFO "postgresql://user:password@example.com/labmate"
#define LABSERV_DB_INIT_SCRIPT_PATH "sql/init.sql"

/* Redis configuration */
#define LABSERV_REDIS_IP "127.0.0.1"
#define LABSERV_REDIS_PORT 6379

/* Logging configuration */
#include "log.h"
#define LOG_LEVEL LOG_VERBOSE

#endif // LABSERV_CONFIG_H