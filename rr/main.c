/*
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    +                                                                              +
    +                            +                                                 +
    +                          +   +                                               +
    +                            +                                                 +
    +                      ++    +    ++                                           +
    +                     +  ++++ ++++  +                                          +
    +                      ++    +    ++          [[==       ==]]                  +  +++++   +++++   +++       +++      ++++           ++++     +++++++            ++++++   + ++++   ++++   +         +      ++++++++ +++++++      
    +                            +                    +     +                      +  +   +   +       +   +      +      +               +   +    +                 +         +         +      +       +      +         +    +      
    +                          +   +                   +   +                       +  +  +    +       +    +     +       +              +   +    +                +           +        +       +     +      +          +   + 
    +        +         +        +                       + +                        +  +++     +++++   +    +    +++       +++           ++ +     ++++++          ++          ++++++    +        =   =      +++++++     + +         
    +       +++       +++                                +                         +  +  +    +       +    +     +           +          +  +     +               +           +         +         + +      +            + +            
    +        +         +                                + +                        +  +   +   +       +   +      +           +          +   +    +               +            +++++    +          =      +             +  +
    +          -------                                 +   +                       +  +    +  +++++   +++       +++     +++++           +    +   +++ ++  ++++     +++++++++           +++                ++++++++      +   +++++
    +           [[[]]                                 +     +                      +
    +          [[[]]]                                +       +                     +
    +            ||              +++             [[==         ==]]                 +
    +        [[ [[]] ]]         +   +                                              +
    +           ****            +   +                                              + 
    +    ^      |()\      ^      +++                                               +
    +   &&&    |()()\    &&&                                                       +
    +    v    |()()()\    v                                                        +
    +    ----|()()()()\----                                                        +
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
*/

#include <hiredis/hiredis.h>
#include <plibsys/plibsys.h>
#include <assert.h>
#include <string.h>
#include "log.h"
#include "mailer.h"
#include "config.h"

#if !defined(RR_REDIS_IP)
#error "RR_REDIS_IP not defined"
#endif
#if !defined(RR_REDIS_PORT)
#error "RR_REDIS_PORT not defined"
#endif

static mailer_t* pRegistrationMailer = NULL;
static redisContext* pRegistrationRedisContext = NULL;

static void main_registration_consumer_init() {
  pRegistrationMailer = mailer_new();
  mailer_config_t mailerConfig = {
    .smtp_user = RR_MAILER_USER,
    .smtp_passwd = RR_MAILER_PASSWD,
    .smtp_server_url = RR_MAILER_SMTP_SERVER_URL,
    .smtp_from_addr = RR_MAILER_SMTP_FROM_ADDR
  };
  LOG_I("main_registration_consumer_init: Initializing registration mailer");
  mailer_init(pRegistrationMailer, &mailerConfig);
  LOG_I("main_registration_consumer_init: Registration mailer initialized");

  LOG_I("main_registration_consumer_init: Connecting to Redis");
  pRegistrationRedisContext = redisConnect(RR_REDIS_IP, RR_REDIS_PORT);
  if (pRegistrationRedisContext == NULL || pRegistrationRedisContext->err) {
    if (pRegistrationRedisContext) {
      LOG_E("main_registration_consumer_init: Failed to connect to Redis: %s", pRegistrationRedisContext->errstr);
      redisFree(pRegistrationRedisContext);
    } else {
      LOG_E("main_registration_consumer_init: Failed to connect to Redis: can't allocate redis context\n");
    }
    exit(EXIT_FAILURE);
  }
  LOG_I("main_registration_consumer_init: Connected to Redis");
}

static void main_registration_consumer_deinit() {
  LOG_I("main_registration_consumer_deinit: Deinitializing and freeing registration mailer");
  mailer_deinit(pRegistrationMailer);
  mailer_free(pRegistrationMailer);
  pRegistrationMailer = NULL;

  LOG_I("main_registration_consumer_deinit: Disconnecting from Redis");
  redisFree(pRegistrationRedisContext);
  pRegistrationRedisContext = NULL;
}

#define REGISTRATION_BASE_URL RR_REGISTRATION_BASE_URL

/**
 * @param req_str Request string in the following format: email|username|verification_code
 * @param recipientAddr_out Output parameter for recipient email address - needs to be freed by caller after use
 */
static char* main_registration_generate_email_body_from_req_str(const char* req_str, char** recipientAddr_out) {
  assert(req_str != NULL);
  const char* emailStart = req_str;
  const char* usernameStart = strchr(emailStart, '|') + 1;
  assert(usernameStart != NULL + 1);
  const char* verificationCodeStart = strchr(usernameStart, '|') + 1;
  assert(verificationCodeStart != NULL + 1);
  assert(strchr(verificationCodeStart, '|') == NULL);

  size_t emailLen = usernameStart - emailStart - 1;
  size_t usernameLen = verificationCodeStart - usernameStart - 1;
  size_t verificationCodeLen = strlen(verificationCodeStart);

  char* email = malloc(emailLen + 1);
  if (email == NULL) {
    LOG_E("main_registration_generate_email_body_from_req_str: Failed to allocate memory for email\n");
    exit(EXIT_FAILURE);
  }
  char* username = malloc(usernameLen + 1);
  if (username == NULL) {
    LOG_E("main_registration_generate_email_body_from_req_str: Failed to allocate memory for username\n");
    free(email);
    exit(EXIT_FAILURE);
  }
  char* verification_code = malloc(verificationCodeLen + 1);
  if (verification_code == NULL) {
    LOG_E("main_registration_generate_email_body_from_req_str: Failed to allocate memory for verification_code\n");
    free(email);
    free(username);
    exit(EXIT_FAILURE);
  }

  strncpy(email, emailStart, emailLen);
  strncpy(username, usernameStart, usernameLen);
  strcpy(verification_code, verificationCodeStart);

  email[emailLen] = '\0';
  username[usernameLen] = '\0';
  verification_code[verificationCodeLen] = '\0';

  const char* emailBodyHello = "Hello ";
  const char* emailBodyThankYou = ", <br>Thank you for registering with us (ęśąćźżń). Click <a href=\"" REGISTRATION_BASE_URL "/api/email-verify?token=";
  //const char* emailBodyThankYou = ", <br>Thank you for registering with us. Click here: \"" REGISTRATION_BASE_URL "/api/email-verify?token=";
  const char* emailBodyUname = "&username=";
  //const char* emailBodyHere = "'>here</a> to verify your email address. <br><br> By proceeding, you agree to the below terms and conditions:<br><br>1. You will not use this service to post any material which is knowingly false and/or defamatory, inaccurate, abusive, vulgar, hateful, harassing, obscene, profane, sexually oriented, threatening, invasive of a person's privacy, or otherwise violative of any law.<br>2. You will not use this service to promote any illegal activities.<br>3. You will not use this service to post any copyrighted material unless the copyright is owned by you.<br><br>";
  //const char* emailBodyHere = "\" to verify your email address. <br><br> By proceeding, you agree to the <i>terms and conditions</i> of LabMate.<br><br>";
  const char* emailBodyHere = "\">here</a> to verify your email address. <br><br> By proceeding, you agree to the <a href=\"" REGISTRATION_BASE_URL "/terms-and-conditions\">Terms and Conditions</a> of LabMate.<br><br>";
  //const char* emailBodyHere = "'>here</a> to verify your email address. <br><br> By proceeding, you agree to the <i>terms and conditions</i> of LabMate.<br><br>";
  //const char* emailBodyHere = "to verify your email address. <br><br> By proceeding, you agree to the <i>terms and conditions</i> of LabMate.<br><br>";
  const char* emailBodyBestRegards = "Best regards,<br>The LabMate Team";
  char* emailBody = malloc(strlen(emailBodyHello) + strlen(username) + strlen(emailBodyThankYou) + strlen(verification_code) + strlen(emailBodyUname) + strlen(username) + strlen(emailBodyHere) + strlen(emailBodyBestRegards) + 1);
  if (emailBody == NULL) {
    LOG_E("main_registration_mail_body_from_request_string: Failed to allocate memory for emailBody\n");
    free(email);
    free(username);
    free(verification_code);
    exit(EXIT_FAILURE);
  }
  strcpy(emailBody, emailBodyHello);
  strcat(emailBody, username);
  strcat(emailBody, emailBodyThankYou);
  strcat(emailBody, verification_code);
  strcat(emailBody, emailBodyUname);
  strcat(emailBody, username);
  strcat(emailBody, emailBodyHere);
  strcat(emailBody, emailBodyBestRegards);

  free(username);
  free(verification_code);
  *recipientAddr_out = email;
  return emailBody;
}

static void main_free_email_body(char* emailBody) {
  assert(emailBody != NULL);
  free(emailBody);
}

#define REGISTRATION_REDIS_QUEUE_NAME RR_REGISTRATION_REDIS_QUEUE_NAME

static void* main_registration_consumer_run(void* pArg) {
  assert(pRegistrationMailer != NULL);
  assert(pRegistrationRedisContext != NULL);
  while (1) {
    LOG_D("main_registration_consumer_run: Executing BLPOP on queue regmail_mq");
    redisReply* pReply = redisCommand(pRegistrationRedisContext, "BLPOP regmail_mq 0"); //bash equivalent: redis-cli BLPOP regmail_mq 0 (0 means wait forever until an element is pushed to the queue)
    if (pReply == NULL) {
      LOG_E("main_registration_consumer_run: Failed to execute BLPOP command\n");
      p_uthread_exit(EXIT_FAILURE);
    }
    LOG_D("main_registration_consumer_run: pReply->type = %d", pReply->type);
    assert(pReply->type == REDIS_REPLY_ARRAY);
    LOG_D("main_registration_consumer_run: pReply->elements = %d", pReply->elements);
    assert(pReply->elements == 2);
    assert(pReply->element[0]->type == REDIS_REPLY_STRING);
    assert(pReply->element[1]->type == REDIS_REPLY_STRING);
    assert(pReply->element[0]->str != NULL);
    assert(!strcmp(pReply->element[0]->str, REGISTRATION_REDIS_QUEUE_NAME));
    assert(pReply->element[1]->str != NULL);
    LOG_I("main_registration_consumer_run: Received registration request: %s", pReply->element[1]->str);
    char* recipientEmailAddr = NULL;
    char* emailBody = main_registration_generate_email_body_from_req_str(pReply->element[1]->str, &recipientEmailAddr);
    mailer_send(pRegistrationMailer, recipientEmailAddr, "LabMate Account Registration Verification Link", emailBody);
    main_free_email_body(emailBody);
    freeReplyObject(pReply);
    free(recipientEmailAddr);
  }
}

int main() {
  p_libsys_init();
  log_global_init();
  main_registration_consumer_init();
  PUThread* pRegistrationThread = p_uthread_create(main_registration_consumer_run, NULL, TRUE, "main_registration_consumer");
  int rv = p_uthread_join(pRegistrationThread);
  assert(-1 != rv);
  LOG_I("main: main_registration_consumer_run exited with exit code %d", rv);
  //mailer_send(pRegistrationMailer, "user1@gmail.com", "Teścik muteksowy 1", "Testowa wiadomość A :)");
  //mailer_send(pRegistrationMailer, "user2@gmail.com", "Teścik muteksowy 2", "Testowa wiadomość B :)");
  main_registration_consumer_deinit();
  log_global_deinit();
  p_libsys_shutdown();
  return 0;
}