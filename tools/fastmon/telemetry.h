#ifndef TELEMETRY_H
#define TELEMETRY_H

#include <sys/socket.h>
#include <arpa/inet.h>

struct telemetry {
  int sockfd;
  struct sockaddr_in servaddr;
  struct sockaddr_in clntaddr;
  int flags;
};
struct telemetry_packet {
  uint8_t rssi0;
  uint8_t rssi1;
  uint8_t epc[12];
} __attribute__((packed));

struct telemetry telemetry_init_client(const char* serverAddr, int port);
struct telemetry telemetry_init_server(int port);
void telemetry_connect(struct telemetry* t);
void telemetry_send(struct telemetry* t, struct telemetry_packet* packet);
void telemetry_receive(struct telemetry* t, struct telemetry_packet* packet);
void telemetry_print_sockopts(struct telemetry* t);
void telemetry_close(struct telemetry* t);

#endif // TELEMETRY_H