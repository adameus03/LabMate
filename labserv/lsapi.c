#include "lsapi.h"
#include <yyjson.h>
#include <bcrypt/bcrypt.h>
#include <hiredis/hiredis.h>
#include <ctype.h>
#include <h2o/websocket.h>
#include "config.h"
#include "log.h"
#include "db.h"

struct lsapi {
    db_t* pDb;
};

lsapi_t* lsapi_new() {
    lsapi_t* pLsapi = (lsapi_t*)malloc(sizeof(lsapi_t));
    if (pLsapi == NULL) {
        LOG_E("lsapi_new: Failed to allocate memory for lsapi_t");
        exit(EXIT_FAILURE);
    }

    return pLsapi;
}

void lsapi_free(lsapi_t* pLsapi) {
    assert(pLsapi != NULL);
    free(pLsapi);
}

void lsapi_init(lsapi_t* pLsapi) {
    assert(pLsapi != NULL);
    pLsapi->pDb = db_new();
    db_init(pLsapi->pDb);
}

void lsapi_deinit(lsapi_t* pLsapi) {
    assert(pLsapi != NULL);
    db_close(pLsapi->pDb);
    db_free(pLsapi->pDb);
}

/**
 * @brief Trick function to obtain lsapi_t* using h2o_handler_t*
 */
static lsapi_t* __lsapi_self_from_h2o_handler(h2o_handler_t* pH2oHandler) {
    assert(pH2oHandler != NULL);
    return (lsapi_t*)*(void**)(pH2oHandler + 1);
}

#define __LSAPI_EMAIL_VERIF_TOKEN_LEN 32

/**
 * @brief Generate a random token for email verification
 * @attention The caller is responsible for freeing the returned token
 */
static char* __lsapi_generate_token() {
    FILE* f = fopen("/dev/urandom", "r");
    if (f == NULL) {
        LOG_E("__lsapi_generate_token: Failed to open /dev/urandom");
        return NULL;
    }
    char* token = (char*)malloc(__LSAPI_EMAIL_VERIF_TOKEN_LEN + 1);
    if (token == NULL) {
        LOG_E("__lsapi_generate_token: Failed to allocate memory for token");
        fclose(f);
        return NULL;
    }
    token[__LSAPI_EMAIL_VERIF_TOKEN_LEN] = '\0';
    size_t result = fread(token, 1, __LSAPI_EMAIL_VERIF_TOKEN_LEN, f);
    if (result != __LSAPI_EMAIL_VERIF_TOKEN_LEN) {
        LOG_E("__lsapi_generate_token: Failure when reading /dev/urandom (fread returned %lu while expecting %d)", result, __LSAPI_EMAIL_VERIF_TOKEN_LEN);
        free(token);
        fclose(f);
        return NULL;
    }
    fclose(f);
    for (int i = 0; i < __LSAPI_EMAIL_VERIF_TOKEN_LEN; i++) {
        token[i] = 'a' + (((unsigned char)token[i]) % 26U);
        assert(token[i] >= 'a' && token[i] <= 'z');
    }
    token[__LSAPI_EMAIL_VERIF_TOKEN_LEN] = '\0';
    return token;
}

#define __LSAPI_REDIS_IP LABSERV_REDIS_IP
#define __LSAPI_REDIS_PORT LABSERV_REDIS_PORT

static void __lsapi_email_push_verification_token(const char* email, const char* username, const char* verification_token) {//TODO
    assert(email != NULL);
    assert(username != NULL);
    assert(verification_token != NULL);
    redisContext* pRedisContext = redisConnect(__LSAPI_REDIS_IP, __LSAPI_REDIS_PORT);
    if (pRedisContext == NULL || pRedisContext->err) {
        if (pRedisContext) {
            LOG_E("__lsapi_email_push_verification_token: Failed to connect to Redis: %s", pRedisContext->errstr);
            redisFree(pRedisContext);
        } else {
            LOG_E("__lsapi_email_push_verification_token: Failed to connect to Redis: can't allocate redis context");
        }
        LOG_W("__lsapi_email_push_verification_token: Failed to push email request for %s (%s) with verification token %s. Not retrying", username, email, verification_token);
        return;
    }
    redisReply* pReply = redisCommand(pRedisContext, "RPUSH regmail_mq %s|%s|%s", email, username, verification_token);
    if (pReply == NULL) {
        LOG_E("__lsapi_email_push_verification_token: Failed to push email request for %s (%s) with verification token %s - redisCommand returned NULL. Not retrying", username, email, verification_token);
        redisFree(pRedisContext);
        return;
    }
    assert(pReply->type == REDIS_REPLY_INTEGER);

    LOG_I("__lsapi_email_push_verification_token: Pushed email request for %s (%s) with verification token %s. Num queued items: %d", username, email, verification_token, (int)(pReply->integer));
    freeReplyObject(pReply);
    redisFree(pRedisContext);
}

static int __Lsapi_endpoint_resp_short(h2o_req_t *pReq, 
                                       const int httpStatus, 
                                       const char* httpReason, 
                                       const char* jsonStatus, 
                                       const char* jsonMessage) {
    assert(pReq != NULL);
    assert(httpReason != NULL);
    assert(jsonStatus != NULL);
    assert(jsonMessage != NULL);
    static h2o_generator_t generator = {NULL, NULL};
    //pReq->res.status = httpStatus;
    pReq->res.status = 200; //TODO remove this, however using a non-200 status code for some reason delays the response and we don't know why, so it needs to be resolved first
    pReq->res.reason = httpReason;
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pJsonRespRoot = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pJsonRespRoot);

    yyjson_mut_obj_add_str(pJsonResp, pJsonRespRoot, "status", jsonStatus);
    yyjson_mut_obj_add_str(pJsonResp, pJsonRespRoot, "message", jsonMessage);
    const char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);
    free((void*)respText);
    yyjson_mut_doc_free(pJsonResp);
    return 0;
}

static int __lsapi_endpoint_error(h2o_req_t *pReq, const int status, const char* reason, const char* errMessage) {    
    // assert(pReq != NULL);
    // assert(reason != NULL);
    // assert(errMessage != NULL);
    // static h2o_generator_t generator = {NULL, NULL};
    // pReq->res.status = status;
    // pReq->res.reason = reason;
    // h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    // h2o_start_response(pReq, &generator);

    // const char* errStatus = "error";
    // yyjson_mut_doc* pJsonErrResp = yyjson_mut_doc_new(NULL);
    // yyjson_mut_val* pRootErrResp = yyjson_mut_obj(pJsonErrResp);
    // yyjson_mut_doc_set_root(pJsonErrResp, pRootErrResp);

    // yyjson_mut_obj_add_str(pJsonErrResp, pRootErrResp, "status", errStatus);
    // yyjson_mut_obj_add_str(pJsonErrResp, pRootErrResp, "message", errMessage);
    // const char* respErrText = yyjson_mut_write(pJsonErrResp, 0, NULL);
    // assert(respErrText != NULL);
    // h2o_iovec_t errBody = h2o_strdup(&pReq->pool, respErrText, SIZE_MAX);
    // h2o_send(pReq, &errBody, 1, 1);
    // free((void*)respErrText);
    // yyjson_mut_doc_free(pJsonErrResp);
    // return 0;
    return __Lsapi_endpoint_resp_short(pReq, status, reason, "error", errMessage);
}

static int __lsapi_endpoint_success(h2o_req_t *pReq, const int status, const char* reason, const char* message) {
    return __Lsapi_endpoint_resp_short(pReq, status, reason, "success", message);
}

static int __lsapi_username_check(const char* username) {
    if (username == NULL) {
        return 0;
    }
    size_t len = strlen(username);
    if (len < 3 || len > 32) {
        return 0;
    }
    for (size_t i = 0; i < len; i++) {
        if (!isalnum(username[i]) && username[i] != '_') {
            return 0;
        }
    }
    return 1;
}

static int __lsapi_endpoint_user_put(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    assert(pLsapi != NULL);
    yyjson_doc* pJson = yyjson_read(pReq->entity.base, pReq->entity.len, 0);
    if (pJson == NULL) {
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid JSON");
    } 
    yyjson_val* pRoot = yyjson_doc_get_root(pJson);
    if (pRoot == NULL || !yyjson_is_obj(pRoot)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing JSON root object");
    }
    yyjson_val* pUsername = yyjson_obj_get(pRoot, "username");
    if (pUsername == NULL || !yyjson_is_str(pUsername)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid username");
    }
    yyjson_val* pEmail = yyjson_obj_get(pRoot, "email");
    if (pEmail == NULL || !yyjson_is_str(pEmail)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid email");
    }
    yyjson_val* pPassword = yyjson_obj_get(pRoot, "password");
    if (pPassword == NULL || !yyjson_is_str(pPassword)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid password");
    }
    yyjson_val* pFirstName = yyjson_obj_get(pRoot, "first_name");
    if (pFirstName == NULL || !yyjson_is_str(pFirstName)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid first name");
    }
    yyjson_val* pLastName = yyjson_obj_get(pRoot, "last_name");
    if (pLastName == NULL || !yyjson_is_str(pLastName)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid last name");
    }
    
    const char* username = yyjson_get_str(pUsername);
    const char* first_name = yyjson_get_str(pFirstName);
    const char* last_name = yyjson_get_str(pLastName);
    const char* email = yyjson_get_str(pEmail);
    const char* password = yyjson_get_str(pPassword);
    
    assert(username != NULL && first_name != NULL && last_name != NULL && email != NULL && password != NULL);

    if (!__lsapi_username_check(username)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid username (required 3-32 characters, alphanumeric and underscores only)");
    }

    // hash password
    char pwd_salt[BCRYPT_HASHSIZE];
    char pwd_hash[BCRYPT_HASHSIZE];
    assert(0 == bcrypt_gensalt(12, pwd_salt));
    assert(0 == bcrypt_hashpw(password, pwd_salt, pwd_hash));
    
    // generate email verification token
    //unsigned char email_verif_token[__LSAPI_EMAIL_VERIF_TOKEN_LEN];
    //randombytes_buf(email_verif_token, __LSAPI_EMAIL_VERIF_TOKEN_LEN);
    //sodium_bin2base64(email_verif_token, __LSAPI_EMAIL_VERIF_TOKEN_LEN, email_verif_token, __LSAPI_EMAIL_VERIF_TOKEN_LEN, sodium_base64_VARIANT_ORIGINAL);
    char* email_verif_token = __lsapi_generate_token();
    if (email_verif_token == NULL) {
        // pReq->res.status = 500;
        // pReq->res.reason = "Internal Server Error";
        // h2o_send_inline(pReq, H2O_STRLIT("Internal Server Error"));
        // yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to generate email verification token. Poor server");
        //return 0;
    }

    //hash email verification token
    char email_verif_token_hash[BCRYPT_HASHSIZE];
    char email_verif_token_salt[BCRYPT_HASHSIZE];
    assert(0 == bcrypt_gensalt(12, email_verif_token_salt));
    assert(0 == bcrypt_hashpw(email_verif_token, email_verif_token_salt, email_verif_token_hash));
    
    assert(email_verif_token_hash[BCRYPT_HASHSIZE - 4] == '\0');
    assert(email_verif_token_salt[(BCRYPT_HASHSIZE - 4)/2 - 1] == '\0');
    assert(strlen(email_verif_token_hash) == BCRYPT_HASHSIZE - 4);
    assert(strlen(email_verif_token_salt) == (BCRYPT_HASHSIZE - 4)/2 - 1);
    LOG_V("__lsapi_endpoint_user_put: email_verif_token: %s, email_verif_token_hash: %s, email_verif_token_salt: %s", email_verif_token, email_verif_token_hash, email_verif_token_salt);

    struct sockaddr sa;
    pReq->conn->callbacks->get_peername(pReq->conn, &sa);
#define __LSAPI_IP_LEN INET6_ADDRSTRLEN > INET_ADDRSTRLEN ? INET6_ADDRSTRLEN : INET_ADDRSTRLEN
    char ip[__LSAPI_IP_LEN];
    memset(ip, 0, __LSAPI_IP_LEN);
    switch(sa.sa_family) {
        case AF_INET:
            inet_ntop(AF_INET, &((struct sockaddr_in*)&sa)->sin_addr, ip, INET_ADDRSTRLEN);
            break;
        case AF_INET6:
            inet_ntop(AF_INET6, &((struct sockaddr_in6*)&sa)->sin6_addr, ip, INET6_ADDRSTRLEN);
            break;
        default:
            break;
    }

    // insert user
    assert(pLsapi->pDb != NULL);
    if (0 != db_user_insert_basic(pLsapi->pDb, username, ip, first_name, last_name, email, pwd_hash, pwd_salt, email_verif_token_hash, email_verif_token_salt)) {
        // TODO check for other errors?
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 409, "Conflict", "Username/email already exists");
    }

    // get user data from database so that we can use it for the http response
    db_user_t user;
    if (0 != db_user_get_by_username(pLsapi->pDb, username, &user)) {
        // pReq->res.status = 500; // Internal Server Error
        // pReq->res.reason = "Internal Server Error";
        // h2o_send_inline(pReq, H2O_STRLIT("Internal Server Error"));
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get user data from database");
        //return 0;
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);
    
    const char* status = "success";
    const char* message = "User created successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add public user data as sub-object
    yyjson_mut_val* pUser = yyjson_mut_obj(pJsonResp);
    yyjson_mut_obj_add_int(pJsonResp, pUser, "user_id", user.user_id);
    yyjson_mut_obj_add_str(pJsonResp, pUser, "username", user.username);
    yyjson_mut_obj_add_str(pJsonResp, pUser, "first_name", user.first_name);
    yyjson_mut_obj_add_str(pJsonResp, pUser, "last_name", user.last_name);
    // add user object to root
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "user", pUser);
    
    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    __lsapi_email_push_verification_token(email, username, email_verif_token);

    free((void*)respText);
    yyjson_doc_free(pJson);
    yyjson_mut_doc_free(pJsonResp);
    free(email_verif_token);
    return 0;
}

static int __lsapi_endpoint_user_get(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    assert(pLsapi != NULL);
    if (pReq->query_at == SIZE_MAX) {
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing query string");
    }
    char* queryStr = pReq->path.base + pReq->query_at + 1;
    size_t queryStrLen = pReq->path.len - pReq->query_at - 1;
    LOG_D("__lsapi_endpoint_user_get: queryStrLen = %lu", queryStrLen);
    LOG_D("__lsapi_endpoint_user_get: queryStr = %.*s", (int)queryStrLen, queryStr);

    const char* userIdParamName = "user_id";
    size_t userIdParamNameLen = strlen(userIdParamName);
    if (queryStrLen < 1) {
        LOG_D("__lsapi_endpoint_user_get: Empty query string (queryStrLen = %lu)", queryStrLen);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Empty query string");
    } else if (queryStrLen < userIdParamNameLen + 2) { // user_id=d is the shortest possible query string (where d is a decimal digit)
        LOG_D("__lsapi_endpoint_user_get: Query string too short (queryStrLen = %lu)", queryStrLen);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Query string too short");
    }
    char* userIdParamNameAddr = strstr(queryStr, userIdParamName);
    if (userIdParamNameAddr == NULL) {
        LOG_D("__lsapi_endpoint_user_get: Missing user_id parameter in query string");
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing user_id parameter in query string");
    } else if (userIdParamNameAddr != queryStr) {
        LOG_D("__lsapi_endpoint_user_get: user_id parameter not at the beginning of query string");
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "user_id parameter not at the beginning of query string");
    }
    char* userIdParamNVSeparatorAddr = userIdParamNameAddr + userIdParamNameLen;
    if (*userIdParamNVSeparatorAddr != '=') {
        LOG_D("__lsapi_endpoint_user_get: Missing = after user_id in query string");
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing = after user_id in query string");
    }
    char* userIdParamValue = userIdParamNVSeparatorAddr + 1;
    size_t userIdParamValueLen = queryStr + queryStrLen - userIdParamValue;
    assert(userIdParamValueLen >= 1);
    for (size_t i = 0; i < userIdParamValueLen; i++) {
        if (!isdigit(userIdParamValue[i])) {
            LOG_D("__lsapi_endpoint_user_get: Invalid user_id value in query string (non-digit character at position %lu in string %.*s)", i, (int)userIdParamValueLen, userIdParamValue);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid user_id value in query string");
        }
    }
    char* userIdParamValueNt = (char*)malloc(userIdParamValueLen + 1);
    if (userIdParamValueNt == NULL) {
        LOG_E("__lsapi_endpoint_user_get: Failed to allocate memory for userIdParamValueNt");
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Server ran out of memory. Poor server.");
    }
    memcpy(userIdParamValueNt, userIdParamValue, userIdParamValueLen);
    userIdParamValueNt[userIdParamValueLen] = '\0';
    int userId = atoi(userIdParamValueNt);
    LOG_V("__lsapi_endpoint_user_get: userId = %d", userId);
    assert(userId >= 0);
    
    // get user data from database so that we can use it for the http response
    db_user_t user; //TODO extract repeated code (with __lsapi_endpoint_user_post) to a separate function in the sake of DRY
    int rv = db_user_get_by_id(pLsapi->pDb, userIdParamValueNt, &user);
    if (0 != rv) {
        if (rv == -2) {
            return __lsapi_endpoint_error(pReq, 404, "Not Found", "User not found");
        } else {
            // pReq->res.status = 500; // Internal Server Error
            // pReq->res.reason = "Internal Server Error";
            // h2o_send_inline(pReq, H2O_STRLIT("Internal Server Error"));
            // return 0;
            LOG_E("__lsapi_endpoint_user_get: Failed to get user data from database (db_user_get_by_id returned %d)", rv);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get user data from database");
        }
    }

    // TODO: Streamline these
    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "User data retrieved successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add public user data as sub-object
    yyjson_mut_val* pUser = yyjson_mut_obj(pJsonResp);
    yyjson_mut_obj_add_int(pJsonResp, pUser, "user_id", user.user_id);
    yyjson_mut_obj_add_str(pJsonResp, pUser, "username", user.username);
    yyjson_mut_obj_add_str(pJsonResp, pUser, "first_name", user.first_name);
    yyjson_mut_obj_add_str(pJsonResp, pUser, "last_name", user.last_name);
    // add user object to root
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "user", pUser);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free(userIdParamValueNt);
    free((void*)respText);
    yyjson_mut_doc_free(pJsonResp);
    return 0;
}

static int __lsapi_endpoint_user_patch(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    assert(pLsapi != NULL);
    LOG_F("__lsapi_endpoint_user_patch: Not implemented"); //TODO Implement when session management is ready
    return __lsapi_endpoint_error(pReq, 501, "Not Implemented", "Not Implemented");
}

// curl -X PUT -d '{"username":"abc","email":"abc@example.com","password":"test","first_name":"test","last_name":"test"}' http://localhost:7890/api/user
int lsapi_endpoint_user(h2o_handler_t* pH2oHandler, h2o_req_t* pReq)
{
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);
    if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("PUT"))) {
        return __lsapi_endpoint_user_put(pH2oHandler, pReq, pLsapi);
    } else if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("GET"))) {
        return __lsapi_endpoint_user_get(pH2oHandler, pReq, pLsapi);
    } else if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("PATCH"))) {
        return __lsapi_endpoint_user_patch(pH2oHandler, pReq, pLsapi);
    } else {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
}

// Request syntax: GET /api/email-verify?token=<token>&username=<username>
int __lsapi_endpoint_email_verify_get(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    assert(pLsapi != NULL);
    //TODO DRY with __lsapi_endpoint_user_get
    if (pReq->query_at == SIZE_MAX) {
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing query string");
    }
    char* queryStr = pReq->path.base + pReq->query_at + 1;
    size_t queryStrLen = pReq->path.len - pReq->query_at - 1;
    LOG_D("__lsapi_endpoint_email_verify_get: queryStrLen = %lu", queryStrLen);
    LOG_D("__lsapi_endpoint_email_verify_get: queryStr = %.*s", (int)queryStrLen, queryStr);

    //TODO Replace with regex if there are benefits
    const char* tokenParamName = "token";
    size_t tokenParamNameLen = strlen(tokenParamName);
    const char* usernameParamName = "username";
    size_t usernameParamNameLen = strlen(usernameParamName);
    if (queryStrLen < 1) {
        LOG_D("__lsapi_endpoint_email_verify_get: Empty query string (queryStrLen = %lu)", queryStrLen);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Empty query string");
    } else if (queryStrLen < tokenParamNameLen + usernameParamNameLen + 5) { // token=x&username=y is the shortest possible query string (where x and y are characters)
        LOG_D("__lsapi_endpoint_email_verify_get: Query string too short (queryStrLen = %lu)", queryStrLen);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Query string too short");
    }
    char* tokenParamNameAddr = strstr(queryStr, tokenParamName);
    if (tokenParamNameAddr == NULL) {
        LOG_D("__lsapi_endpoint_email_verify_get: Missing token parameter in query string");
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing token parameter in query string");
    } else if (tokenParamNameAddr != queryStr) {
        LOG_D("__lsapi_endpoint_email_verify_get: token parameter not at the beginning of query string");
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "token parameter not at the beginning of query string");
    }
    char* tokenParamNVSeparatorAddr = tokenParamNameAddr + tokenParamNameLen;
    if (*tokenParamNVSeparatorAddr != '=') {
        LOG_D("__lsapi_endpoint_email_verify_get: Missing = after token in query string");
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing = after token in query string");
    }
    char* tokenParamValue = tokenParamNVSeparatorAddr + 1;
    char* tokenUsernameSeparatorAddr = strchr(tokenParamValue, '&');
    if (tokenUsernameSeparatorAddr == NULL) {
        LOG_D("__lsapi_endpoint_email_verify_get: Missing & after token in query string");
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing & after token in query string");
    }
    size_t tokenParamValueLen = tokenUsernameSeparatorAddr - tokenParamValue;
    if (tokenParamValueLen < 1) { // example: curl -X GET 'http://localhost:7890/api/email-verify?token=&swefrebterfqwgefrgr'
        LOG_D("__lsapi_endpoint_email_verify_get: Empty token parameter value in query string");
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Empty token parameter value in query string");
    }

    char* usernameParamNameAddr = strstr(tokenUsernameSeparatorAddr, usernameParamName);
    if (usernameParamNameAddr == NULL) {
        LOG_D("__lsapi_endpoint_email_verify_get: Missing username parameter in query string");
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing username parameter in query string");
    } else if (usernameParamNameAddr != tokenUsernameSeparatorAddr + 1) {
        LOG_D("__lsapi_endpoint_email_verify_get: username parameter not after & in query string");
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "username parameter not after & in query string");
    }
    char* usernameParamNVSeparatorAddr = usernameParamNameAddr + usernameParamNameLen;
    if (*usernameParamNVSeparatorAddr != '=') {
        LOG_D("__lsapi_endpoint_email_verify_get: Missing = after username in query string");
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing = after username in query string");
    }
    char* usernameParamValue = usernameParamNVSeparatorAddr + 1;
    if (NULL != strchr(usernameParamValue, '&')) {
        LOG_D("__lsapi_endpoint_email_verify_get: Extra & after username in query string");
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Extra & after username in query string");
    }
    size_t usernameParamValueLen = queryStr + queryStrLen - usernameParamValue;
    if (usernameParamValueLen < 1) { //Shouldn't happen like in the token case as extra & detection would be triggered, however we are defensive
        LOG_D("__lsapi_endpoint_email_verify_get: Empty username parameter value in query string");
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Empty username parameter value in query string");
    }

    char* tokenParamValueNt = (char*)malloc(tokenParamValueLen + 1);
    if (tokenParamValueNt == NULL) {
        LOG_E("__lsapi_endpoint_email_verify_get: Failed to allocate memory for tokenParamValueNt");
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Server ran out of memory. Poor server.");
    }
    memcpy(tokenParamValueNt, tokenParamValue, tokenParamValueLen);
    tokenParamValueNt[tokenParamValueLen] = '\0';
    char* usernameParamValueNt = (char*)malloc(usernameParamValueLen + 1);
    if (usernameParamValueNt == NULL) {
        LOG_E("__lsapi_endpoint_email_verify_get: Failed to allocate memory for usernameParamValueNt");
        free(tokenParamValueNt);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Server ran out of memory. Poor server.");
    }
    memcpy(usernameParamValueNt, usernameParamValue, usernameParamValueLen);
    usernameParamValueNt[usernameParamValueLen] = '\0';

    // get user data from database so that we can verify the token
    db_user_t user;
    int rv = db_user_get_by_username(pLsapi->pDb, usernameParamValueNt, &user);
    if (0 != rv) {
        if (rv == -2) {
            free(tokenParamValueNt);
            free(usernameParamValueNt);
            return __lsapi_endpoint_error(pReq, 404, "Not Found", "User not found");
        } else {
            free(tokenParamValueNt);
            free(usernameParamValueNt);
            LOG_E("__lsapi_endpoint_email_verify_get: Failed to get user data from database (db_user_get_by_username returned %d)", rv);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get user data from database");
        }
    }
    assert(user.email_verification_token_hash[BCRYPT_HASHSIZE - 4] == '\0');
    assert(user.email_verification_token_salt[(BCRYPT_HASHSIZE - 4)/2 - 1] == '\0');
    assert(strlen(user.email_verification_token_hash) == BCRYPT_HASHSIZE - 4);
    assert(strlen(user.email_verification_token_salt) == (BCRYPT_HASHSIZE - 4)/2 - 1);
    
    // verify token
    char userProvidedTokenHash[BCRYPT_HASHSIZE];
    assert(user.email_verification_token_hash != NULL);
    assert(user.email_verification_token_salt != NULL);
    assert(0 == bcrypt_hashpw(tokenParamValueNt, user.email_verification_token_salt, userProvidedTokenHash)); //TODO What about bcrypt_checkpw? (I don't know why it doesn't accept salt as additional argument like `bcrypt_hashpw` - we'd need to figure that out)

    assert(userProvidedTokenHash[BCRYPT_HASHSIZE - 4] == '\0');
    assert(strlen(userProvidedTokenHash) == BCRYPT_HASHSIZE - 4);
    LOG_V("__lsapi_endpoint_email_verify_get: tokenParamValueNt: %s", tokenParamValueNt);
    LOG_V("__lsapi_endpoint_email_verify_get: userProvidedTokenHash: %s, user.email_verification_token_hash: %s, user.email_verification_token_salt: %s", userProvidedTokenHash, user.email_verification_token_hash, user.email_verification_token_salt);

    //if (0 != memcmp(userProvidedTokenHash, user.email_verification_token_hash, BCRYPT_HASHSIZE)) {
    if (0 != strcmp(userProvidedTokenHash, user.email_verification_token_hash)) {
        free(tokenParamValueNt);
        free(usernameParamValueNt);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Invalid token");
    }

    // Mark user as email-verified
    if (0 != db_user_set_email_verified(pLsapi->pDb, user.username)) {
        free(tokenParamValueNt);
        free(usernameParamValueNt);
        LOG_E("__lsapi_endpoint_email_verify_get: Failed to update user data in database while trying to mark user as email-verified");
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to update user data in database");
    }

    free(tokenParamValueNt);
    free(usernameParamValueNt);

    //Redirect to email-verification.html (which automatically redirects to login.html after a timeout)
    pReq->res.status = 303;
    pReq->res.reason = "See Other";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_LOCATION, NULL, H2O_STRLIT("/email-verification.html"));
    h2o_send_inline(pReq, H2O_STRLIT("Redirecting to email-verification.html"));

    return 0;
}

int lsapi_endpoint_email_verify(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);
    if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("GET"))) {
        return __lsapi_endpoint_email_verify_get(pH2oHandler, pReq, pLsapi);
    } else {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
}

int lsapi_endpoint_service_status(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);

    //return __lsapi_endpoint_success(pReq, 200, "OK", "Service is running");

    if (!h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("GET"))) {
        //return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }

    // //test timeout
    // for (int i=0; i<5; i++){
    //     LOG_V("lsapi_endpoint_service_status: Sleeping for 1 second (iteration %d of 5)", i+1);
    //     sleep(1);
    // }

    return __lsapi_endpoint_success(pReq, 200, "OK", "Service is running");
}

// curl -X PUT -d '{"username":"abc","password":"test"}' http://localhost:7890/api/session
int __lsapi_endpoint_session_put(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    assert(pLsapi != NULL);
    yyjson_doc* pJson = yyjson_read(pReq->entity.base, pReq->entity.len, 0);
    if (pJson == NULL) {
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid JSON");
    } 
    yyjson_val* pRoot = yyjson_doc_get_root(pJson);
    if (pRoot == NULL || !yyjson_is_obj(pRoot)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing JSON root object");
    }
    yyjson_val* pUsername = yyjson_obj_get(pRoot, "username");
    if (pUsername == NULL || !yyjson_is_str(pUsername)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid username");
    }
    yyjson_val* pPassword = yyjson_obj_get(pRoot, "password");
    if (pPassword == NULL || !yyjson_is_str(pPassword)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid password");
    }
    
    const char* username = yyjson_get_str(pUsername);
    const char* password = yyjson_get_str(pPassword);
    assert(username != NULL && password != NULL);

    // get user data from database so that we can verify the password
    db_user_t user;
    int rv = db_user_get_by_username(pLsapi->pDb, username, &user);
    if (0 != rv) {
        if (rv == -2) {
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 404, "Not Found", "User not found");
        } else {
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get user data from database");
        }
    }

    // verify password
    char userProvidedPwdHash[BCRYPT_HASHSIZE];
    assert(user.passwd_hash != NULL);
    assert(user.passwd_salt != NULL);
    assert(0 == bcrypt_hashpw(password, user.passwd_salt, userProvidedPwdHash));
    assert(userProvidedPwdHash[BCRYPT_HASHSIZE - 4] == '\0');
    assert(strlen(userProvidedPwdHash) == BCRYPT_HASHSIZE - 4);
    LOG_V("__lsapi_endpoint_session_put: user-provided password: %s", password);
    LOG_V("__lsapi_endpoint_session_put: userProvidedPwdHash: %s, user.passwd_hash: %s, user.passwd_salt: %s", userProvidedPwdHash, user.passwd_hash, user.passwd_salt);

    if (0 != strcmp(userProvidedPwdHash, user.passwd_hash)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Invalid password");
    }
    // If user is not email-verified, we don't allow login
    if (!user.is_email_verified) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Email not verified");
    }

    // generate session token
    char* session_token = __lsapi_generate_token();
    if (session_token == NULL) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to generate session token. Poor server");
    }

    //hash session token
    char session_token_hash[BCRYPT_HASHSIZE];
    char session_token_salt[BCRYPT_HASHSIZE];
    assert(0 == bcrypt_gensalt(12, session_token_salt));
    assert(0 == bcrypt_hashpw(session_token, session_token_salt, session_token_hash));

    assert(session_token_hash[BCRYPT_HASHSIZE - 4] == '\0');
    assert(session_token_salt[(BCRYPT_HASHSIZE - 4)/2 - 1] == '\0');
    assert(strlen(session_token_hash) == BCRYPT_HASHSIZE - 4);
    assert(strlen(session_token_salt) == (BCRYPT_HASHSIZE - 4)/2 - 1);
    LOG_V("__lsapi_endpoint_session_put: session_token: %s, session_token_hash: %s, session_token_salt: %s", session_token, session_token_hash, session_token_salt);

    // set session credentials in database
    if (0 != db_user_set_session(pLsapi->pDb, username, session_token_hash, session_token_salt)) {
        free(session_token);
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to set session credentials in database");
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Session created successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add public user data as sub-object
    yyjson_mut_val* pUser = yyjson_mut_obj(pJsonResp);
    yyjson_mut_obj_add_int(pJsonResp, pUser, "user_id", user.user_id);
    yyjson_mut_obj_add_str(pJsonResp, pUser, "username", user.username);
    yyjson_mut_obj_add_str(pJsonResp, pUser, "first_name", user.first_name);
    yyjson_mut_obj_add_str(pJsonResp, pUser, "last_name", user.last_name);
    // add user object to root
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "user", pUser);
    // add session key
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "session_key", session_token);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_doc_free(pJson);
    yyjson_mut_doc_free(pJsonResp);
    free(session_token);
    return 0;
}

// curl -X DELETE -d '{"username":"abc", "session_key":"woedimioduoid"}' http://localhost:7890/api/session
int __lsapi_endpoint_session_delete(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    assert(pLsapi != NULL);
    yyjson_doc* pJson = yyjson_read(pReq->entity.base, pReq->entity.len, 0);
    if (pJson == NULL) {
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid JSON");
    }
    yyjson_val* pRoot = yyjson_doc_get_root(pJson);
    if (pRoot == NULL || !yyjson_is_obj(pRoot)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing JSON root object");
    }
    yyjson_val* pUsername = yyjson_obj_get(pRoot, "username");
    if (pUsername == NULL || !yyjson_is_str(pUsername)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid username");
    }
    yyjson_val* pSessionKey = yyjson_obj_get(pRoot, "session_key");
    if (pSessionKey == NULL || !yyjson_is_str(pSessionKey)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid session_key");
    }

    const char* username = yyjson_get_str(pUsername);
    const char* session_key = yyjson_get_str(pSessionKey);
    assert(username != NULL && session_key != NULL);

    // get user data from database so that we can verify the session key
    db_user_t user;
    int rv = db_user_get_by_username(pLsapi->pDb, username, &user);
    if (0 != rv) {
        if (rv == -2) {
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 404, "Not Found", "User not found");
        } else {
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get user data from database");
        }
    }

    // verify session key
    char userProvidedSessionKeyHash[BCRYPT_HASHSIZE];
    assert(user.sesskey_hash != NULL);
    assert(user.sesskey_salt != NULL);
    assert(0 == bcrypt_hashpw(session_key, user.sesskey_salt, userProvidedSessionKeyHash));
    assert(userProvidedSessionKeyHash[BCRYPT_HASHSIZE - 4] == '\0');
    assert(strlen(userProvidedSessionKeyHash) == BCRYPT_HASHSIZE - 4);
    LOG_V("__lsapi_endpoint_session_delete: user-provided session_key: %s", session_key);
    LOG_V("__lsapi_endpoint_session_delete: userProvidedSessionKeyHash: %s, user.sesskey_hash: %s, user.sesskey_salt: %s", userProvidedSessionKeyHash, user.sesskey_hash, user.sesskey_salt);

    if (0 != strcmp(userProvidedSessionKeyHash, user.sesskey_hash)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Invalid session key");
    }

    // delete session credentials in database
    if (0 != db_user_unset_session(pLsapi->pDb, username)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to delete session credentials in database");
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Session deleted successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    
    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_doc_free(pJson);
    yyjson_mut_doc_free(pJsonResp);
    return 0;
}

int lsapi_endpoint_session(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);
    if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("PUT"))) {
        return __lsapi_endpoint_session_put(pH2oHandler, pReq, pLsapi);
    } else if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("DELETE"))) {
        return __lsapi_endpoint_session_delete(pH2oHandler, pReq, pLsapi);
    } else {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
}

// TODO replace with actual implementation (As it's a simple echo endpoint for now)
static void __lsapi_endpoint_ws_on_msg(h2o_websocket_conn_t* pWsConn, const struct wslay_event_on_msg_recv_arg* pArg) {
    assert(pWsConn != NULL);
    if (pArg == NULL) {
        h2o_websocket_close(pWsConn);
        return;
    }
    if (!wslay_is_ctrl_frame(pArg->opcode)) {
        struct wslay_event_msg msgarg = {pArg->opcode, pArg->msg, pArg->msg_length};
        wslay_event_queue_msg(pWsConn->ws_ctx, &msgarg);
    }   
}

int lsapi_endpoint_ws(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    const char* client_key = NULL;
    int rv = h2o_is_websocket_handshake(pReq, &client_key);
    if (rv != 0) {
        LOG_W("lsapi_endpoint_ws: h2o_is_websocket_handshake failed with rv = %d", rv);
        return -1;
    }
    if (client_key == NULL) { // Prevent crash when endpoint is accessed in a wrong way
        LOG_W("lsapi_endpoint_ws: client_key is NULL");
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing client key");
    }
    h2o_upgrade_to_websocket(pReq, client_key, NULL, __lsapi_endpoint_ws_on_msg);
    return 0;
}