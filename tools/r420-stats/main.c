#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <signal.h>
#include "r420.h"

#define READER_IP_ADDR "192.168.18.43"
#define READER_PORT 5084
// #define READER_IP_ADDR "127.0.0.1"
// #define READER_PORT 4000

void print_header(const r420_msg_hdr_t *pHdr) {
  printf("Message header:\n");
  printf("  Reserved: 0x%X\n", R420_MSG_HDR_RESERVED(*pHdr));
  printf("  Version: 0x%X\n", R420_MSG_HDR_VERSION(*pHdr));
  printf("  Message type: 0x%X\n", R420_MSG_HDR_MESSAGE_TYPE(*pHdr));
  printf("  Message length: %u\n", R420_MSG_HDR_MSG_LENGTH(*pHdr));
  printf("  Message ID: %u\n", R420_MSG_HDR_MSG_ID(*pHdr));
}

uint32_t get_ip_addr(const char *ip_str) {
  unsigned int b1, b2, b3, b4;
  assert(4 == sscanf(ip_str, "%u.%u.%u.%u", &b1, &b2, &b3, &b4));
  assert(b1 <= 255 && b2 <= 255 && b3 <= 255 && b4 <= 255);
  return (b1 << 24) | (b2 << 16) | (b3 << 8) | b4;
}

int should_stop = 0;

static void signal_handler(int sig){
    if (sig == SIGINT) {
      printf("Caught SIGINT, preparing to stop...\n");
      should_stop = 1;
    }
}

#define STOP_ITERATION_NO 30
void main_r420_loop_handler(const r420_ctx_t *pCtx) {
  uint32_t *pCounter = (uint32_t *)pCtx->pUserData;
  (*pCounter)++;
  //printf("main_r420_loop_handler: Iteration %u\n", *pCounter);
  // if (*pCounter >= 6) {
  //   printf("main_r420_loop_handler: Reached %u iterations, exiting loop.\n", *pCounter);
  //   r420_unloop((r420_ctx_t *)pCtx);
  //   printf("main_r420_loop_handler: Exited loop.\n");
  // }

  //if(*pCounter == STOP_ITERATION_NO || should_stop) {
  if (should_stop) {
    //printf("main_r420_loop_handler: Reached iteration %u, calling r420_stop().\n", *pCounter);
    r420_stop((r420_ctx_t *)pCtx);
  }
}

// printf("Measurement count: %d\n", ++counter);
//     return 0;

int main_log_filter(const char *pMsg) {
  static int counter = 0;
  return 1;
  if (strstr(pMsg, "EPC:") == NULL) {
    return 0;
  }
  return 1;
  // if (strstr(pMsg, "BBBBBBBBBBBBBBBBBBBBBBBB") != NULL) {
  //   return 1;
  // }
  // if (strstr(pMsg, "CCCCCCCCCCCCCCCCCCCCCCCC") != NULL) {
  //   return 1;
  // }
  // if (strstr(pMsg, "DDDDDDDDDDDDDDDDDDDDDDDD") != NULL) {
  //   return 1;
  // }

  // if (strstr(pMsg, "AAAAAAAAAAAAAAAAAAAAAAAA") != NULL) {
  //   return 1;
  // }
  // if (strstr(pMsg, "30396062C39662C0C12707A5") != NULL) {
  //   return 1;
  // }
  return 0;
}

void main_r420_log_handler(const r420_ctx_t *pCtx, const char *pMsg) {
  if (!main_log_filter(pMsg)) {
    return;
  }
  printf("R420 Log: %s\n", pMsg);
  // 
}

int main(int argc, char **argv) {
  signal(SIGINT, signal_handler);
  setvbuf(stdout, NULL, _IOLBF, 0);  // line-buffered stdout
  r420_connection_parameters_t conn_params = {
    .ip = get_ip_addr(READER_IP_ADDR),
    .port = READER_PORT
  };
  printf("Connecting to reader at %s:%u...\n", READER_IP_ADDR, READER_PORT);
  r420_ctx_t ctx = r420_connect(conn_params);
  printf("Connected.\n");
  // printf("Receiving message header...\n");
  // r420_msg_hdr_t hdr = r420_receive_header(&ctx);
  // printf("Received message header.\n");
  // printf("Printing message header:\n");
  // printf("Raw data:\n");
  // const uint8_t *pHdrBytes = (const uint8_t *)&hdr;
  // for (size_t i = 0; i < sizeof(hdr); i++) {
  //   printf(" %02X", pHdrBytes[i]);
  // }
  // printf("\n");
  // print_header(&hdr);
  // printf("Terminating connection.\n");
  // r420_close(&ctx);
  uint32_t r420_loop_counter = 0;
  r420_ctx_set_loop_handler(&ctx, main_r420_loop_handler);
  r420_ctx_set_log_handler(&ctx, main_r420_log_handler);
  printf("Entering r420 loop...\n");
  r420_loop(&ctx, &r420_loop_counter);
  printf("Exited r420 loop after %u iterations.\n", r420_loop_counter);
  printf("Terminating connection.\n");
  r420_close(&ctx);
  printf("Terminated.\n");

  return 0;
}