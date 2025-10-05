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

int main(int argc, char **argv) {
  r420_connection_parameters_t conn_params = {
    .ip = get_ip_addr(READER_IP_ADDR),
    .port = READER_PORT
  };
  printf("Connecting to reader at %s:%u...\n", READER_IP_ADDR, READER_PORT);
  r420_ctx_t ctx = r420_connect(conn_params);
  printf("Connected.\n");
  printf("Receiving message header...\n");
  r420_msg_hdr_t hdr = r420_receive_header(&ctx);
  printf("Received message header.\n");
  printf("Printing message header:\n");
  printf("Raw data:\n");
  const uint8_t *pHdrBytes = (const uint8_t *)&hdr;
  for (size_t i = 0; i < sizeof(hdr); i++) {
    printf(" %02X", pHdrBytes[i]);
  }
  printf("\n");
  print_header(&hdr);
  printf("Terminating connection.\n");
  r420_close(&ctx);
  return 0;
}