#include "mailer.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <plibsys/plibsys.h>
#include "log.h"
#include <curl/curl.h>

struct mailer {
  CURL* pCurl;
  struct curl_slist* pRecipients;
  struct curl_readfunction_param {
    const char* generatedText;
    int bytesRemaining;
  } curlReadFunctionParam;
  const char* smtp_from_addr;
  PMutex* pMutex; // For concurrent `mailer_send` calls
};

mailer_t* mailer_new() {
  mailer_t* pMailer = malloc(sizeof(mailer_t));
  if (pMailer == NULL) {
    LOG_E("mailer_new: Failed to allocate memory for mailer_t\n");
    exit(EXIT_FAILURE);
  }
  return pMailer;
}
void mailer_free(mailer_t* pMailer) {
  assert(pMailer != NULL);
  free(pMailer);
}

// See https://curl.se/libcurl/c/CURLOPT_READFUNCTION.html for documentation of the callback for the CURLOPT_READFUNCTION option
static size_t curl_readfunction(void* buffer, size_t size, size_t nitems, void* userdata) {
  assert(buffer != NULL);
  assert(size == 1);
  struct curl_readfunction_param* pParam = (struct curl_readfunction_param*)userdata;
  assert(pParam != NULL);
  assert(pParam->generatedText != NULL);
  assert(pParam->bytesRemaining == -1 || pParam->bytesRemaining == 0); // 0 means that the generatedText has been read, -1 means that it has not been read at all yet. We don't support partial reads.
  if (pParam->bytesRemaining == 0) {
    pParam->bytesRemaining = -1;
    return 0;
  }
  size_t len = strlen(pParam->generatedText);
  assert(size * nitems >= len);
  memcpy(buffer, pParam->generatedText, len);
  pParam->bytesRemaining = 0;
  return len;
}

void mailer_init(mailer_t* pMailer, mailer_config_t* pConfig) {
  assert(pMailer != NULL);
  assert(pConfig != NULL);
  assert(pConfig->smtp_user != NULL);
  assert(pConfig->smtp_passwd != NULL);
  assert(pConfig->smtp_server_url != NULL);
  assert(pConfig->smtp_from_addr != NULL);
  pMailer->pCurl = curl_easy_init();
  if (pMailer->pCurl == NULL) {
    LOG_E("mailer_init: Failed to initialize curl\n");
    exit(EXIT_FAILURE);
  }
  curl_easy_setopt(pMailer->pCurl, CURLOPT_USERNAME, pConfig->smtp_user);
  curl_easy_setopt(pMailer->pCurl, CURLOPT_PASSWORD, pConfig->smtp_passwd);
  curl_easy_setopt(pMailer->pCurl, CURLOPT_URL, pConfig->smtp_server_url);
  curl_easy_setopt(pMailer->pCurl, CURLOPT_MAIL_FROM, pConfig->smtp_from_addr);
  pMailer->pRecipients = NULL;
  curl_easy_setopt(pMailer->pCurl, CURLOPT_MAIL_RCPT, pMailer->pRecipients);
  curl_easy_setopt(pMailer->pCurl, CURLOPT_READFUNCTION, curl_readfunction);
  pMailer->smtp_from_addr = pConfig->smtp_from_addr;
  pMailer->curlReadFunctionParam.generatedText = NULL;
  pMailer->curlReadFunctionParam.bytesRemaining = -1;
  curl_easy_setopt(pMailer->pCurl, CURLOPT_READDATA, &pMailer->curlReadFunctionParam);
  curl_easy_setopt(pMailer->pCurl, CURLOPT_UPLOAD, 1L);
  curl_easy_setopt(pMailer->pCurl, CURLOPT_VERBOSE, 1L);

  pMailer->pMutex = p_mutex_new();
  if (pMailer->pMutex == NULL) {
    LOG_E("mailer_init: Failed to initialize pMutex\n");
    exit(EXIT_FAILURE);
  }
}

void mailer_deinit(mailer_t* pMailer) {
  assert(pMailer != NULL);
  assert(pMailer->pMutex != NULL);
  assert(pMailer->pCurl != NULL);

  assert(pMailer->pRecipients == NULL);
  assert(pMailer->curlReadFunctionParam.generatedText == NULL);

  p_mutex_free(pMailer->pMutex);
  pMailer->pMutex = NULL;

  curl_easy_cleanup(pMailer->pCurl);
  pMailer->pCurl = NULL;
}

static void mailer_generate_text(mailer_t* pMailer, const char* from, const char* to, const char* subject, const char* body) {
  assert(pMailer != NULL);
  assert(from != NULL);
  assert(to != NULL);
  assert(subject != NULL);
  assert(body != NULL);
  // header = prolog + value + epilog
  const char* toProlog = "To: ";
  const char* toEpilog = "\r\n";
  const char* fromProlog = "From: ";
  const char* fromEpilog = "\r\n";
  const char* mimeVersion = "MIME-Version: 1.0\r\n";
  const char* contentType = "Content-Type: text/html; charset=utf-8\r\n";
  //const char* contentTransferEncoding = "Content-Transfer-Encoding: quoted-printable\r\n";
  const char* subjectProlog = "Subject: ";
  const char* subjectEpilog = "\r\n\r\n";
  char* generatedText = malloc(strlen(toProlog) + strlen(to) + strlen(toEpilog) + strlen(fromProlog) + strlen(from) + strlen(fromEpilog) + strlen(mimeVersion) + strlen(contentType) /*+ strlen(contentTransferEncoding)*/ + strlen(subjectProlog) + strlen(subject) + strlen(subjectEpilog) + strlen(body) + 1);
  if (generatedText == NULL) {
    LOG_E("mailer_generate_text: Failed to allocate memory for generatedText\n");
    exit(EXIT_FAILURE);
  }
  strcpy(generatedText, toProlog);
  strcat(generatedText, to);
  strcat(generatedText, toEpilog);
  strcat(generatedText, fromProlog);
  strcat(generatedText, from);
  strcat(generatedText, fromEpilog);
  strcat(generatedText, mimeVersion);
  strcat(generatedText, contentType);
  //strcat(generatedText, contentTransferEncoding);
  strcat(generatedText, subjectProlog);
  strcat(generatedText, subject);
  strcat(generatedText, subjectEpilog);
  strcat(generatedText, body);
  assert(pMailer->curlReadFunctionParam.generatedText == NULL);
  pMailer->curlReadFunctionParam.generatedText = generatedText;
}

static void mailer_free_text(mailer_t* pMailer) {
  assert(pMailer != NULL);
  free((void*)pMailer->curlReadFunctionParam.generatedText);
  pMailer->curlReadFunctionParam.generatedText = NULL;
}

void mailer_send(mailer_t* pMailer, const char* to, const char* subject, const char* body) {
  assert(pMailer != NULL);
  assert(pMailer->pMutex != NULL);
  assert(TRUE == p_mutex_lock(pMailer->pMutex)); // Only one `mailer_send` call can be in progress at a time

  assert(to != NULL);
  assert(subject != NULL);
  assert(body != NULL);
  assert(pMailer->curlReadFunctionParam.generatedText == NULL);
  mailer_generate_text(pMailer, pMailer->smtp_from_addr, to, subject, body);
  assert(pMailer->pRecipients == NULL);
  pMailer->pRecipients = curl_slist_append(pMailer->pRecipients, to);
  assert(pMailer->pRecipients != NULL);
  curl_easy_setopt(pMailer->pCurl, CURLOPT_MAIL_RCPT, pMailer->pRecipients);
  CURLcode res = curl_easy_perform(pMailer->pCurl);
  if (res != CURLE_OK) {
    LOG_W("mailer_send: Failed to send email to %s. Not retrying. Error message was: %s, error code was: %d\n", to, curl_easy_strerror(res), res);
  } else {
    LOG_I("mailer_send: Email sent successfully to %s\n", to);
  }
  assert(pMailer->pRecipients != NULL);
  curl_slist_free_all(pMailer->pRecipients);
  pMailer->pRecipients = NULL;
  assert(pMailer->curlReadFunctionParam.generatedText != NULL);
  mailer_free_text(pMailer);

  assert(TRUE == p_mutex_unlock(pMailer->pMutex));
}
