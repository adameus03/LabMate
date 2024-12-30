/*
  IMPORTANT: This is a sample configuration file. Before building the project, you should copy this file to config.h and modify it to suit your needs.
*/

#ifndef RR_CONFIG_H
#define RR_CONFIG_H

#define RR_REDIS_IP "127.0.0.1"
#define RR_REDIS_PORT 6379

#define RR_MAILER_SMTP_SERVER_URL "smtps://smtp.example.com:465"
#define RR_MAILER_SMTP_FROM_ADDR "noreply@example.com"
#define RR_MAILER_USER "user"
#define RR_MAILER_PASSWD "password"

#define RR_REGISTRATION_BASE_URL "https://labmate.example.com"
#define RR_REGISTRATION_REDIS_QUEUE_NAME "regmail_mq"

#endif // RR_CONFIG_H