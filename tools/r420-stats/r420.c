#include <sys/socket.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <assert.h>
#include "r420.h"

r420_ctx_t r420_connect(const r420_connection_parameters_t conn_params) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  assert(fd > 0);
  struct sockaddr_in addr = {
    .sin_family = AF_INET,
    .sin_port = htons(conn_params.port),  // Convert to network byte order
    .sin_addr.s_addr = htonl(conn_params.ip)  // Convert to network byte order
  };
  assert(0 == connect(fd, (struct sockaddr *)&addr, sizeof(addr)));
  return (r420_ctx_t){ .fd = fd };
}

void r420_close(r420_ctx_t *pCtx) {
  assert(0 == close(pCtx->fd));
  pCtx->fd = -1;
}

r420_msg_hdr_t r420_receive_header(const r420_ctx_t *pCtx) {
  r420_msg_hdr_t hdr = {0};
  ssize_t remaining_rcv_bytes = sizeof(hdr);
  while (remaining_rcv_bytes > 0) {
    ssize_t n = read(pCtx->fd, &hdr, sizeof(hdr));
    assert(n > 0);
    remaining_rcv_bytes -= n;
  }
  return hdr;
}

void r420_loop(const r420_ctx_t *pCtx, r420_loop_callback_t loop_handler) {
  while (1) {
    r420_msg_hdr_t hdr = r420_receive_header(pCtx);
    loop_handler(pCtx, &hdr);
  }
}