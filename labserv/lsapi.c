#include "lsapi.h"
#include <yyjson.h>

int lsapi_endpoint_main(h2o_handler_t *self, h2o_req_t *req)
{
    if (h2o_memis(req->method.base, req->method.len, H2O_STRLIT("POST"))) {
        static h2o_generator_t generator = {NULL, NULL};
        req->res.status = 200;
        req->res.reason = "OK";
        h2o_add_header(&req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("text/plain; charset=utf-8"));
        h2o_start_response(req, &generator);
        h2o_iovec_t body = h2o_strdup(&req->pool, "api response\n", SIZE_MAX);
        h2o_send(req, &body, 1, 1);
        return 0;
    }

    return -1;
}