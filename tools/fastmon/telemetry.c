#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h> // bzero()
#include <unistd.h> // read(), write(), close()
#include <assert.h>
#include "telemetry.h"

struct telemetry telemetry_init_client(const char* serverAddr, int port) {
    struct telemetry t;
    t.sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (t.sockfd == -1) {
        printf("socket creation failed...\n");
        exit(0);
    }
    else
        printf("Socket successfully created..\n");
    bzero(&t.servaddr, sizeof(t.servaddr));
    t.servaddr.sin_family = AF_INET;
    t.servaddr.sin_addr.s_addr = inet_addr(serverAddr);
    t.servaddr.sin_port = htons(port);
    t.flags = 0; // client
    return t;
}

struct telemetry telemetry_init_server(int port) {
    struct telemetry t;
    t.sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (t.sockfd == -1) {
        printf("socket creation failed...\n");
        exit(0);
    }
    else
        printf("Socket successfully created..\n");
    bzero(&t.servaddr, sizeof(t.servaddr));
    t.servaddr.sin_family = AF_INET;
    t.servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    t.servaddr.sin_port = htons(port);
    if ((bind(t.sockfd, (struct sockaddr*)&t.servaddr, sizeof(t.servaddr))) != 0) {
        printf("socket bind failed...\n");
        exit(0);
    }
    else
        printf("Socket successfully binded..\n");
    if ((listen(t.sockfd, 1)) != 0) {
        printf("Listen failed...\n");
        exit(0);
    }
    else
        printf("Server listening..\n");
    t.flags = 1; // server
    return t;
}

static void telemetry_connect_client(struct telemetry* t) {
    if (connect(t->sockfd, (struct sockaddr*)&t->servaddr, sizeof(t->servaddr))
        != 0) {
        printf("connection with the server failed...\n");
        exit(0);
    }
    else
        printf("connected to the server..\n");
}

static void telemetry_connect_server(struct telemetry* t) {
    socklen_t len = sizeof(t->clntaddr);
    t->sockfd = accept(t->sockfd, (struct sockaddr*)&t->clntaddr, &len);
    if (t->sockfd < 0) {
        printf("server acccept failed...\n");
        exit(0);
    }
    else
        printf("server acccept the client...\n");
}

void telemetry_connect(struct telemetry* t) {
    switch (t->flags) {
    case 0:
        telemetry_connect_client(t);
        break;
    case 1:
        telemetry_connect_server(t);
        break;
    default:
        printf("invalid flags...\n");
        exit(0);
    }
}

void telemetry_send(struct telemetry* t, struct telemetry_packet* packet) {
    assert(sizeof(struct telemetry_packet) == 14);
    int n = write(t->sockfd, packet, sizeof(struct telemetry_packet));
    if (n != sizeof(struct telemetry_packet)) {
        printf("write failed...\n");
        exit(0);
    }
}

void telemetry_receive(struct telemetry* t, struct telemetry_packet* packet) {
    int n = read(t->sockfd, packet, sizeof(struct telemetry_packet));
    if (n != sizeof(struct telemetry_packet)) {
        printf("read failed...\n");
        exit(0);
    }
}

void telemetry_print_sockopts(struct telemetry* t) {
    //print socket optioins
    int optval;
    socklen_t optlen = sizeof(optval);
    int rv = getsockopt(t->sockfd, SOL_SOCKET, SO_RCVBUF, &optval, &optlen);
    if (rv != 0) {
        printf("getsockopt failed...\n");
        exit(0);
    }
    printf("SO_RCVBUF: %d\n", optval);
    rv = getsockopt(t->sockfd, SOL_SOCKET, SO_SNDBUF, &optval, &optlen);
    if (rv != 0) {
        printf("getsockopt failed...\n");
        exit(0);
    }
    printf("SO_SNDBUF: %d\n", optval);
}

void telemetry_close(struct telemetry* t) {
    int rv = close(t->sockfd);
    if (rv != 0) {
        printf("close failed...\n");
        exit(0);
    }
}
