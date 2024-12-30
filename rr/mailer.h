#ifndef MAILER_H
#define MAILER_H

/**
 * @file mailer.h
 * @brief Provides a simple interface to send emails using libcurl behind the scenes.
 */

typedef struct mailer mailer_t;
typedef struct mailer_config {
  const char* smtp_user;
  const char* smtp_passwd;
  const char* smtp_server_url;
  const char* smtp_from_addr;
} mailer_config_t;

// Allocates memory for a new mailer_t instance and returns a pointer to it (thread-safe)
mailer_t* mailer_new();
// Frees the memory allocated for a mailer_t instance (thread-safe)
void mailer_free(mailer_t* pMailer);
// Initializes a mailer_t instance with the given configuration (thread-safe, but only for different pMailer instances)
void mailer_init(mailer_t* pMailer, mailer_config_t* pConfig);
// Deinitializes a mailer_t instance (thread-safe, but only for different pMailer instances)
void mailer_deinit(mailer_t* pMailer);
// Sends an email with the given subject and body to the given recipient (thread-safe, even for the same `mailer_t` instance - however each thread will block until the previous thread has finished sending the email, UNLESS you use different `mailer_t` instances)
void mailer_send(mailer_t* pMailer, const char* to, const char* subject, const char* body);


#endif // MAILER_H