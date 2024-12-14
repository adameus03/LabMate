#ifndef LSAPI_H
#define LSAPI_H

#include <h2o.h>

int lsapi_endpoint_main(h2o_handler_t *self, h2o_req_t *req);

#endif //LSAPI_H