/*
 * Copyright (c) 2014 DeNA Co., Ltd.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include <errno.h>
#include <limits.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include "h2o.h"
#include "h2o/http1.h"
#include "h2o/http2.h"
#include "h2o/memcached.h"
#include <plibsys/plibsys.h>

#include "config.h"
#include "lsapi.h"

struct main_endpoint_handler_spawn_params {
    h2o_handler_t* pH2oHandler;
    h2o_req_t* pReq;
};

static struct main_endpoint_handler_spawn_params* main_endpoint_handler_spawn_params_new(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    struct main_endpoint_handler_spawn_params* pParams = (struct main_endpoint_handler_spawn_params*)malloc(sizeof(struct main_endpoint_handler_spawn_params));
    pParams->pH2oHandler = pH2oHandler;
    pParams->pReq = pReq;
    return pParams;
}

static void main_endpoint_handler_spawn_params_free(struct main_endpoint_handler_spawn_params* pParams) {
    assert(pParams != NULL);
    free(pParams);
}

static void main_endpoint_handler_spawn_handler(void* pUserData) {
    struct main_endpoint_handler_spawn_params* pParams = (struct main_endpoint_handler_spawn_params*)pUserData;
    int (*on_req)(h2o_handler_t *, h2o_req_t *) = NULL;
    on_req = *((void**)(pParams->pH2oHandler + 2));
    on_req(pParams->pH2oHandler, pParams->pReq);
    main_endpoint_handler_spawn_params_free(pParams); //TODO For some reason this causes a failed assertion in h2o when compiled with asan. Investigate (the failed assertion is: ./lib/common/socket/uv-binding.c.h:139: do_write: Assertion `sock->super._cb.write == NULL' failed., backtrace points to h2o_socket_write)
}

static int main_endpoint_handler_spawn(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    uv_thread_t uvThreadId;
    struct main_endpoint_handler_spawn_params* pParams = main_endpoint_handler_spawn_params_new(pH2oHandler, pReq);
    
#if LABSERV_ENDPOINTS_THREADING_ENABLED == 1
    assert(0 == uv_thread_create(&uvThreadId, main_endpoint_handler_spawn_handler, (void*)pParams));
#else
    main_endpoint_handler_spawn_handler((void*)pParams);
#endif
    //p_uthread_create(main_endpoint_handler_spawn_handler, (void*)pParams, FALSE, NULL);
}

static h2o_pathconf_t *register_handler(h2o_hostconf_t *hostconf, const char *path, int (*on_req)(h2o_handler_t *, h2o_req_t *), void* pUserData, h2o_access_log_filehandle_t* pLogFh)
{
    assert(sizeof(h2o_handler_t) > sizeof(void*));
    assert(sizeof(h2o_handler_t) % sizeof(void*) == 0); //defensive?

    h2o_pathconf_t *pathconf = h2o_config_register_path(hostconf, path, 0);
    //h2o_handler_t* pHandler = h2o_create_handler(pathconf, sizeof(h2o_handler_t) + 2 * sizeof(void*));
    h2o_handler_t* pHandler = h2o_create_handler(pathconf, 3 * sizeof(h2o_handler_t)); // `3 * sizeof(h2o_handler_t)` not `sizeof(h2o_handler_t) + 2 * sizeof(void*)` to avoid memory bug (it worked with glibc, but segfaulted with musl libc)

    *((void**)(pHandler + 1)) = pUserData;
    *((void**)(pHandler + 2)) = on_req;
    //pHandler->on_req = on_req;
    pHandler->on_req = main_endpoint_handler_spawn;

    if (pLogFh != NULL) {
        h2o_access_log_register(pathconf, pLogFh);
    }
    return pathconf;
}

static h2o_globalconf_t config;
static h2o_context_t ctx;
static h2o_multithread_receiver_t libmemcached_receiver;
static h2o_accept_ctx_t accept_ctx;

#if H2O_USE_LIBUV

static void on_accept(uv_stream_t *listener, int status)
{
    uv_tcp_t *conn;
    h2o_socket_t *sock;

    if (status != 0)
        return;

    conn = h2o_mem_alloc(sizeof(*conn));
    uv_tcp_init(listener->loop, conn);

    if (uv_accept(listener, (uv_stream_t *)conn) != 0) {
        uv_close((uv_handle_t *)conn, (uv_close_cb)free);
        return;
    }

    sock = h2o_uv_socket_create((uv_stream_t *)conn, (uv_close_cb)free);
    h2o_accept(&accept_ctx, sock);
}

static int create_listener(void)
{
    static uv_tcp_t listener;
    struct sockaddr_in addr;
    int r;

    uv_tcp_init(ctx.loop, &listener);
    uv_ip4_addr(LABSERV_IPV4_ADDR, LABSERV_IP_PORT, &addr);
    if ((r = uv_tcp_bind(&listener, (struct sockaddr *)&addr, 0)) != 0) {
        fprintf(stderr, "uv_tcp_bind:%s\n", uv_strerror(r));
        goto Error;
    }
    if ((r = uv_listen((uv_stream_t *)&listener, 128, on_accept)) != 0) {
        fprintf(stderr, "uv_listen:%s\n", uv_strerror(r));
        goto Error;
    }

    return 0;
Error:
    uv_close((uv_handle_t *)&listener, NULL);
    return r;
}

#else

static void on_accept(h2o_socket_t *listener, const char *err)
{
    h2o_socket_t *sock;

    if (err != NULL) {
        return;
    }

    if ((sock = h2o_evloop_socket_accept(listener)) == NULL)
        return;
    h2o_accept(&accept_ctx, sock);
}

static int create_listener(void)
{
    struct sockaddr_in addr;
    int fd, reuseaddr_flag = 1;
    h2o_socket_t *sock;

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(0x7f000001);
    addr.sin_port = htons(LABSERV_IP_PORT);

    if ((fd = socket(AF_INET, SOCK_STREAM, 0)) == -1 ||
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuseaddr_flag, sizeof(reuseaddr_flag)) != 0 ||
        bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0 || listen(fd, SOMAXCONN) != 0) {
        return -1;
    }

    sock = h2o_evloop_socket_create(ctx.loop, fd, H2O_SOCKET_FLAG_DONT_READ);
    h2o_socket_read_start(sock, on_accept);

    return 0;
}

#endif

static int setup_ssl(const char *cert_file, const char *key_file, const char *ciphers)
{
    SSL_load_error_strings();
    SSL_library_init();
    OpenSSL_add_all_algorithms();

    accept_ctx.ssl_ctx = SSL_CTX_new(SSLv23_server_method());
    SSL_CTX_set_options(accept_ctx.ssl_ctx, SSL_OP_NO_SSLv2);

    if (LABSERV_USE_MEMCACHED) {
        accept_ctx.libmemcached_receiver = &libmemcached_receiver;
        //h2o_accept_setup_memcached_ssl_resumption(h2o_memcached_create_context(LABSERV_IPV4_ADDR, 11211, 0, 1, "h2o:ssl-resumption:"), 86400);
        h2o_accept_setup_async_ssl_resumption(h2o_memcached_create_context(LABSERV_IPV4_ADDR, 11211, 0, 1, "h2o:ssl-resumption:"), 86400);
        h2o_socket_ssl_async_resumption_setup_ctx(accept_ctx.ssl_ctx);
    }

#ifdef SSL_CTX_set_ecdh_auto
    SSL_CTX_set_ecdh_auto(accept_ctx.ssl_ctx, 1);
#endif

    /* load certificate and private key */
    if (SSL_CTX_use_certificate_chain_file(accept_ctx.ssl_ctx, cert_file) != 1) {
        fprintf(stderr, "an error occurred while trying to load server certificate file:%s\n", cert_file);
        return -1;
    }
    if (SSL_CTX_use_PrivateKey_file(accept_ctx.ssl_ctx, key_file, SSL_FILETYPE_PEM) != 1) {
        fprintf(stderr, "an error occurred while trying to load private key file:%s\n", key_file);
        return -1;
    }

    if (SSL_CTX_set_cipher_list(accept_ctx.ssl_ctx, ciphers) != 1) {
        fprintf(stderr, "ciphers could not be set: %s\n", ciphers);
        return -1;
    }

/* setup protocol negotiation methods */
#if H2O_USE_NPN
    h2o_ssl_register_npn_protocols(accept_ctx.ssl_ctx, h2o_http2_npn_protocols);
#endif
#if H2O_USE_ALPN
    h2o_ssl_register_alpn_protocols(accept_ctx.ssl_ctx, h2o_http2_alpn_protocols);
#endif

    return 0;
}

int main(int argc, char **argv)
{
    p_libsys_init();
    log_global_init();
    lsapi_t* pLsapi = lsapi_new();
    lsapi_init(pLsapi);

    h2o_hostconf_t *hostconf;
    h2o_access_log_filehandle_t *logfh = h2o_access_log_open_handle(LABSERV_H2O_ACCESS_LOG_FILE_PATH, NULL, H2O_LOGCONF_ESCAPE_APACHE);
    h2o_pathconf_t *pathconf;

    signal(SIGPIPE, SIG_IGN);

    h2o_config_init(&config);
    hostconf = h2o_config_register_host(&config, h2o_iovec_init(H2O_STRLIT("default")), 65535);
    
    /***************************/
    /* Endpoint registrations */
    /*************************/
    pathconf = register_handler(hostconf, "/api/user", lsapi_endpoint_user, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/email-verify", lsapi_endpoint_email_verify, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/service-status", lsapi_endpoint_service_status, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/session", lsapi_endpoint_session, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/ws", lsapi_endpoint_ws, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/invm", lsapi_endpoint_invm, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/invm-bulk", lsapi_endpoint_invm_bulk, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/inventory", lsapi_endpoint_inventory, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/inventory-items", lsapi_endpoint_inventory_items, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/inven-ld", lsapi_endpoint_inven_ld, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/lab", lsapi_endpoint_lab, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/labs", lsapi_endpoint_labs, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/reagent", lsapi_endpoint_reagent, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/reagents", lsapi_endpoint_reagents, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/vendor", lsapi_endpoint_vendor, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/vendors", lsapi_endpoint_vendors, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/reagtype", lsapi_endpoint_reagtype, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/reagtypes", lsapi_endpoint_reagtypes, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/faculty", lsapi_endpoint_faculty, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/faculties", lsapi_endpoint_faculties, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/antenna", lsapi_endpoint_antenna, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/localize", lsapi_endpoint_localize, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/basepoint", lsapi_endpoint_basepoint, (void*)pLsapi, logfh);
    pathconf = register_handler(hostconf, "/api/localization-results", lsapi_endpoint_localization_results, (void*)pLsapi, logfh);

    pathconf = h2o_config_register_path(hostconf, "/", 0);
    h2o_file_register(pathconf, "htdocs", NULL, NULL, 0);
    if (logfh != NULL)
        h2o_access_log_register(pathconf, logfh);

#if H2O_USE_LIBUV
    uv_loop_t loop;
    uv_loop_init(&loop);
    h2o_context_init(&ctx, &loop, &config);
#else
    h2o_context_init(&ctx, h2o_evloop_create(), &config);
#endif
    if (LABSERV_USE_MEMCACHED)
        h2o_multithread_register_receiver(ctx.queue, &libmemcached_receiver, h2o_memcached_receiver);

    if (LABSERV_USE_HTTPS && setup_ssl("examples/h2o/server.crt", "examples/h2o/server.key",
                               "DEFAULT:!MD5:!DSS:!DES:!RC4:!RC2:!SEED:!IDEA:!NULL:!ADH:!EXP:!SRP:!PSK") != 0)
        goto Error;

    accept_ctx.ctx = &ctx;
    accept_ctx.hosts = config.hosts;

    if (create_listener() != 0) {
        fprintf(stderr, "failed to listen to %s:%d:%s\n", LABSERV_IPV4_ADDR, LABSERV_IP_PORT, strerror(errno));
        goto Error;
    }

#if H2O_USE_LIBUV
    uv_run(ctx.loop, UV_RUN_DEFAULT);
#else
    while (h2o_evloop_run(ctx.loop, INT32_MAX) == 0)
        ;
#endif

Error:
    lsapi_deinit(pLsapi);
    lsapi_free(pLsapi);
    log_global_deinit();
    p_libsys_shutdown();
    return 1;
}