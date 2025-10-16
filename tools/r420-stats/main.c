#include <stdio.h>
#include <assert.h>
#include "r420.h"

#define READER_IP_ADDR "192.168.18.43"
#define READER_PORT 5084

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


void main_r420_loop_handler(const r420_ctx_t *pCtx) {
  uint32_t *pCounter = (uint32_t *)pCtx->pUserData;
  (*pCounter)++;
  printf("main_r420_loop_handler: Iteration %u\n", *pCounter);
  if (*pCounter >= 2) {
    printf("main_r420_loop_handler: Reached %u iterations, exiting loop.\n", *pCounter);
    r420_unloop((r420_ctx_t *)pCtx);
    printf("main_r420_loop_handler: Exited loop.\n");
  }
}

void main_r420_log_handler(const r420_ctx_t *pCtx, const char *pMsg) {
  printf("R420 Log: %s\n", pMsg);
}

int main(int argc, char **argv) {
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