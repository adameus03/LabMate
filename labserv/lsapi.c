#include "lsapi.h"
#include <yyjson.h>
#include <bcrypt/bcrypt.h>
#include <hiredis/hiredis.h>
#include <ctype.h>
#include <h2o/websocket.h>
#include <plibsys/plibsys.h>
#include <curl/curl.h>
#include "config.h"
#include "log.h"
#include "db.h"
#include "oph.h"

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

static void __lsapi_email_push_verification_token(const char* email, const char* username, const char* verification_token) {
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

static int __lsapi_endpoint_resp_short(h2o_req_t *pReq, 
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
    return __lsapi_endpoint_resp_short(pReq, status, reason, "error", errMessage);
}

static int __lsapi_endpoint_success(h2o_req_t *pReq, const int status, const char* reason, const char* message) {
    return __lsapi_endpoint_resp_short(pReq, status, reason, "success", message);
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

#define __LSAPI_IP_LEN (INET6_ADDRSTRLEN > INET_ADDRSTRLEN ? INET6_ADDRSTRLEN : INET_ADDRSTRLEN)

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
    db_user_free(&user);
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
    db_user_free(&user);
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
static int __lsapi_endpoint_email_verify_get(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Invalid token");
    }

    // Mark user as email-verified
    if (0 != db_user_set_email_verified(pLsapi->pDb, user.username)) {
        free(tokenParamValueNt);
        free(usernameParamValueNt);
        db_user_free(&user);
        LOG_E("__lsapi_endpoint_email_verify_get: Failed to update user data in database while trying to mark user as email-verified");
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to update user data in database");
    }

    free(tokenParamValueNt);
    free(usernameParamValueNt);
    db_user_free(&user);

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
static int __lsapi_endpoint_session_put(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Invalid password");
    }
    // If user is not email-verified, we don't allow login
    if (!user.is_email_verified) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Email not verified");
    }

    // generate session token
    char* session_token = __lsapi_generate_token();
    if (session_token == NULL) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
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
        db_user_free(&user);
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
    db_user_free(&user);
    return 0;
}

// curl -X DELETE -d '{"username":"abc", "session_key":"woedimioduoid"}' http://localhost:7890/api/session
static int __lsapi_endpoint_session_delete(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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

    if (0 == strlen(user.sesskey_hash)) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Session key not set");
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
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Invalid session key");
    }

    // delete session credentials in database
    if (0 != db_user_unset_session(pLsapi->pDb, username)) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
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
    db_user_free(&user);
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

// curl -X PUT -d '{"rtname":"abc", "username":"abc", "session_key":"<sesskey>"}' http://localhost:7890/api/reagtype
static int __lsapi_endpoint_reagtype_put(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
    yyjson_val* pRtName = yyjson_obj_get(pRoot, "rtname");
    if (pRtName== NULL || !yyjson_is_str(pRtName)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rtname (reagent type name)");
    }
    yyjson_val* pUsername = yyjson_obj_get(pRoot, "username");
    if (pUsername == NULL || !yyjson_is_str(pUsername)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid username");
    }
    yyjson_val* pSessionKey = yyjson_obj_get(pRoot, "session_key");
    if (pSessionKey == NULL || !yyjson_is_str(pSessionKey)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid session_key in request body");
    }

    const char* rtName = yyjson_get_str(pRtName);
    const char* username = yyjson_get_str(pUsername);
    const char* userProvidedSessionKey = yyjson_get_str(pSessionKey);

    assert(rtName != NULL && username != NULL && userProvidedSessionKey != NULL);

    // TODO replace repeating code with a separate function
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

    if (0 == strlen(user.sesskey_hash)) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Session key not set");
    }

    // verify session key
    char userProvidedSessionKeyHash[BCRYPT_HASHSIZE];
    assert(user.sesskey_hash != NULL);
    assert(strlen(user.sesskey_hash) == BCRYPT_HASHSIZE - 4);
    assert(user.sesskey_salt != NULL);
    assert(strlen(user.sesskey_salt) == (BCRYPT_HASHSIZE - 4)/2 - 1);

    assert(0 == bcrypt_hashpw(userProvidedSessionKey, user.sesskey_salt, userProvidedSessionKeyHash));
    assert(userProvidedSessionKeyHash[BCRYPT_HASHSIZE - 4] == '\0');
    assert(strlen(userProvidedSessionKeyHash) == BCRYPT_HASHSIZE - 4);
    LOG_V("__lsapi_endpoint_reagtype_put: user-provided session key: %s", userProvidedSessionKey);
    LOG_V("__lsapi_endpoint_reagtype_put: userProvidedSessionKeyHash: %s, user.sesskey_hash: %s, user.sesskey_salt: %s", userProvidedSessionKeyHash, user.sesskey_hash, user.sesskey_salt);

    if (0 != strcmp(userProvidedSessionKeyHash, user.sesskey_hash)) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Invalid session key");
    }

    // create reagent type + get reagent type data from database so that we can use it for the http response
    db_reagent_type_t reagent_type;
    if (0 != db_reagent_type_insert_ret(pLsapi->pDb, rtName, &reagent_type)) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to create reagent type");
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Reagent type created successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add reagent type data as sub-object
    yyjson_mut_val* pReagentType = yyjson_mut_obj(pJsonResp);
    yyjson_mut_obj_add_int(pJsonResp, pReagentType, "reagtype_id", reagent_type.reagtype_id);
    yyjson_mut_obj_add_str(pJsonResp, pReagentType, "name", reagent_type.name);
    // add reagent type object to root
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "reagtype", pReagentType);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_doc_free(pJson);
    yyjson_mut_doc_free(pJsonResp);
    db_user_free(&user);
    db_reagent_type_free(&reagent_type);
    return 0;
}

// TODO use it for user and reagent too for consistency (reagent is in enum, but not used in __lsapi_endpoint_reagent_get for now)
typedef enum __lsapi_x_type {
    __LSAPI_X_TYPE_REAGTYPE,
    __LSAPI_X_TYPE_REAGENT,
    __LSAPI_X_TYPE_VENDOR,
    __LSAPI_X_TYPE_FACULTY,
    __LSAPI_X_TYPE_LAB,
    __LSAPI_X_TYPE_INVENTORY_ITEM,
    __LSAPI_X_TYPE_ANTENNA,

    __LSAPI_X_TYPE__FIRST = __LSAPI_X_TYPE_REAGTYPE,
    __LSAPI_X_TYPE__LAST = __LSAPI_X_TYPE_ANTENNA
} __lsapi_x_type_t;

static int __lsapi_endpoint_x_get(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi, __lsapi_x_type_t xType) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    assert(pLsapi != NULL);
    assert(xType >= __LSAPI_X_TYPE__FIRST && xType <= __LSAPI_X_TYPE__LAST);

    const char* xName = NULL;
    switch (xType) {
        case __LSAPI_X_TYPE_REAGTYPE:
            xName = "reagtype";
            break;
        case __LSAPI_X_TYPE_REAGENT:
            xName = "reagent";
            break;
        case __LSAPI_X_TYPE_VENDOR:
            xName = "vendor";
            break;
        case __LSAPI_X_TYPE_FACULTY:
            xName = "faculty";
            break;
        case __LSAPI_X_TYPE_LAB:
            xName = "lab";
            break;
        case __LSAPI_X_TYPE_INVENTORY_ITEM:
            xName = "inventory_item";
            break;
        case __LSAPI_X_TYPE_ANTENNA:
            xName = "antenna";
            break;
        default:
            assert(0);
    }

    if (pReq->query_at == SIZE_MAX) {
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing query string");
    }
    char* queryStr = pReq->path.base + pReq->query_at + 1;
    size_t queryStrLen = pReq->path.len - pReq->query_at - 1;
    LOG_D("__lsapi_endpoint_x_get: queryStrLen = %lu", queryStrLen);
    LOG_D("__lsapi_endpoint_x_get: queryStr = %.*s", (int)queryStrLen, queryStr);

    size_t xNameLen = strlen(xName);
    char* xIdParamNameEpilog = "_id";
    size_t xIdParamNameEpilogLen = strlen(xIdParamNameEpilog);
    char* xIdParamName = (char*)malloc(xNameLen + xIdParamNameEpilogLen + 1);
    if (xIdParamName == NULL) {
        LOG_E("__lsapi_endpoint_x_get: Failed to allocate memory for xIdParamName");
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Server ran out of memory. Poor server.");
    }
    memcpy(xIdParamName, xName, xNameLen);
    memcpy(xIdParamName + xNameLen, xIdParamNameEpilog, xIdParamNameEpilogLen);
    xIdParamName[xNameLen + xIdParamNameEpilogLen] = '\0';
    size_t xIdParamNameLen = strlen(xIdParamName);
    assert(xIdParamNameLen == xNameLen +xIdParamNameEpilogLen);
    if (queryStrLen < 1) {
        LOG_D("__lsapi_endpoint_x_get: Empty query string (queryStrLen = %lu)", queryStrLen);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Empty query string");
    } else if (queryStrLen < xIdParamNameLen + 2) { // x_id=d is the shortest possible query string (where d is a decimal digit)
        LOG_D("__lsapi_endpoint_reagent_get: Query string too short (queryStrLen = %lu)", queryStrLen);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Query string too short");
    }
    char* xIdParamNameAddr = strstr(queryStr, xIdParamName);
    if (xIdParamNameAddr == NULL) {
        LOG_D("__lsapi_endpoint_x_get: Missing x_id parameter in query string");
        char* errMsgProlog = "Missing ";
        char* errMsgEpilog = " parameter in query string";
        size_t errMsgPrologStrlen = strlen(errMsgProlog);
        size_t errMsgEpilogStrlen = strlen(errMsgEpilog);
        char* errMsg = (char*)malloc(errMsgPrologStrlen + xIdParamNameLen + errMsgEpilogStrlen + 1);
        if (errMsg == NULL) {
            LOG_E("__lsapi_endpoint_x_get: Failed to allocate memory for errMsg");
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Server ran out of memory. Poor server.");
        }
        memcpy(errMsg, errMsgProlog, errMsgPrologStrlen);
        memcpy(errMsg + errMsgPrologStrlen, xIdParamName, xIdParamNameLen);
        memcpy(errMsg + errMsgPrologStrlen + xIdParamNameLen, errMsgEpilog, errMsgEpilogStrlen);
        errMsg[errMsgPrologStrlen + xIdParamNameLen + errMsgEpilogStrlen] = '\0';
        int rv = __lsapi_endpoint_error(pReq, 400, "Bad Request", errMsg);
        free(errMsg);
        return rv;
    } else if (xIdParamNameAddr != queryStr) {
        LOG_D("__lsapi_endpoint_x_get: x_id parameter not at the beginning of query string");
        char* errMsgEpilog = " parameter not at the beginning of query string";
        size_t errMsgEpilogStrlen = strlen(errMsgEpilog);
        char* errMsg = (char*)malloc(xIdParamNameLen + errMsgEpilogStrlen + 1);
        if (errMsg == NULL) {
            LOG_E("__lsapi_endpoint_x_get: Failed to allocate memory for errMsg");
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Server ran out of memory. Poor server.");
        }
        memcpy(errMsg, xIdParamName, xIdParamNameLen);
        memcpy(errMsg + xIdParamNameLen, errMsgEpilog, errMsgEpilogStrlen);
        errMsg[xIdParamNameLen + errMsgEpilogStrlen] = '\0';
        int rv = __lsapi_endpoint_error(pReq, 400, "Bad Request", errMsg);
        free(errMsg);
        return rv;
    }
    char* xIdParamNVSeparatorAddr = xIdParamNameAddr + xIdParamNameLen; // obtains Name-Value separator address
    if (*xIdParamNVSeparatorAddr != '=') {
        LOG_D("__lsapi_endpoint_x_get: Missing = after x_id in query string");
        char* errMsgProlog = "Missing = after ";
        char* errMsgEpilog = " in query string";
        size_t errMsgPrologStrlen = strlen(errMsgProlog);
        size_t errMsgEpilogStrlen = strlen(errMsgEpilog);
        char* errMsg = (char*)malloc(errMsgPrologStrlen + xIdParamNameLen + errMsgEpilogStrlen + 1);
        if (errMsg == NULL) {
            LOG_E("__lsapi_endpoint_x_get: Failed to allocate memory for errMsg");
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Server ran out of memory. Poor server.");
        }
        memcpy(errMsg, errMsgProlog, errMsgPrologStrlen);
        memcpy(errMsg + errMsgPrologStrlen, xIdParamName, xIdParamNameLen);
        memcpy(errMsg + errMsgPrologStrlen + xIdParamNameLen, errMsgEpilog, errMsgEpilogStrlen);
        errMsg[errMsgPrologStrlen + xIdParamNameLen + errMsgEpilogStrlen] = '\0';
        int rv = __lsapi_endpoint_error(pReq, 400, "Bad Request", errMsg);
        free(errMsg);
        return rv;
    }
    char* xIdParamValue = xIdParamNVSeparatorAddr + 1;
    size_t xIdParamValueLen = queryStr + queryStrLen - xIdParamValue;
    assert(xIdParamValueLen >= 1);
    for (size_t i = 0; i < xIdParamValueLen; i++) {
        if (!isdigit(xIdParamValue[i])) {
            LOG_D("__lsapi_endpoint_x_get: Invalid x_id value in query string (non-digit character at position %lu in string %.*s)", i, (int)xIdParamValueLen, xIdParamValue);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid x_id value in query string");
        }
    }
    char* xIdParamValueNt = (char*)malloc(xIdParamValueLen + 1); // xIdParamValueNt = xIdParamValue Null-terminated
    if (xIdParamValueNt == NULL) {
        LOG_E("__lsapi_endpoint_x_get: Failed to allocate memory for xIdParamValueNt");
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Server ran out of memory. Poor server.");
    }
    memcpy(xIdParamValueNt, xIdParamValue, xIdParamValueLen);
    xIdParamValueNt[xIdParamValueLen] = '\0';
    int xId = atoi(xIdParamValueNt);
    LOG_V("__lsapi_endpoint_reagent_get: xId = %d", xId);
    assert(xId >= 0);

    // get x data from database so that we can use it for the http response
    void* pX = NULL;
    int rv = -1;
    switch (xType) {
        case __LSAPI_X_TYPE_REAGTYPE:
            pX = (void*)malloc(sizeof(db_reagent_type_t));
            rv = db_reagent_type_get_by_id(pLsapi->pDb, xIdParamValueNt, (db_reagent_type_t*)pX);
            break;
        case __LSAPI_X_TYPE_REAGENT:
            pX = (void*)malloc(sizeof(db_reagent_t));
            rv = db_reagent_get_by_id(pLsapi->pDb, xIdParamValueNt, (db_reagent_t*)pX);
            break;
        case __LSAPI_X_TYPE_VENDOR:
            pX = (void*)malloc(sizeof(db_vendor_t));
            rv = db_vendor_get_by_id(pLsapi->pDb, xIdParamValueNt, (db_vendor_t*)pX);
            break;
        case __LSAPI_X_TYPE_FACULTY:
            pX = (void*)malloc(sizeof(db_faculty_t));
            rv = db_faculty_get_by_id(pLsapi->pDb, xIdParamValueNt, (db_faculty_t*)pX);
            break;
        case __LSAPI_X_TYPE_LAB:
            pX = (void*)malloc(sizeof(db_lab_t));
            rv = db_lab_get_by_id(pLsapi->pDb, xIdParamValueNt, (db_lab_t*)pX);
            break;
        case __LSAPI_X_TYPE_INVENTORY_ITEM:
            pX = (void*)malloc(sizeof(db_inventory_item_t));
            rv = db_inventory_get_by_id(pLsapi->pDb, xIdParamValueNt, (db_inventory_item_t*)pX);
            break;
        case __LSAPI_X_TYPE_ANTENNA:
            pX = (void*)malloc(sizeof(db_antenna_t));
            rv = db_antenna_get_by_id(pLsapi->pDb, xIdParamValueNt, (db_antenna_t*)pX);
            break;
        default:
            assert(0);
    }
    if (0 != rv) {
        if (rv == -2) {
            char* errMsgEpilog = " not found";
            size_t errMsgEpilogStrlen = strlen(errMsgEpilog);
            char* errMsg = (char*)malloc(xNameLen + errMsgEpilogStrlen + 1);
            if (errMsg == NULL) {
                LOG_E("__lsapi_endpoint_x_get: Failed to allocate memory for errMsg");
                return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Server ran out of memory. Poor server.");
            }
            memcpy(errMsg, xName, xNameLen);
            memcpy(errMsg + xNameLen, errMsgEpilog, errMsgEpilogStrlen);
            errMsg[xNameLen + errMsgEpilogStrlen] = '\0';
            int rv = __lsapi_endpoint_error(pReq, 404, "Not Found", errMsg);
            free(errMsg);
            return rv;
        } else {
            LOG_E("__lsapi_endpoint_x_get: Failed to get x data from database (db_%s_get_by_id returned %d)", xName, rv);
            char* errMsgProlog = "Failed to get ";
            char* errMsgEpilog = " data from database";
            size_t errMsgPrologStrlen = strlen(errMsgProlog);
            size_t errMsgEpilogStrlen = strlen(errMsgEpilog);
            char* errMsg = (char*)malloc(errMsgPrologStrlen + xNameLen + errMsgEpilogStrlen + 1);
            if (errMsg == NULL) {
                LOG_E("__lsapi_endpoint_x_get: Failed to allocate memory for errMsg");
                return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Server ran out of memory. Poor server.");
            }
            memcpy(errMsg, errMsgProlog, errMsgPrologStrlen);
            memcpy(errMsg + errMsgPrologStrlen, xName, xNameLen);
            memcpy(errMsg + errMsgPrologStrlen + xNameLen, errMsgEpilog, errMsgEpilogStrlen);
            errMsg[errMsgPrologStrlen + xNameLen + errMsgEpilogStrlen] = '\0';
            int rv = __lsapi_endpoint_error(pReq, 500, "Internal Server Error", errMsg);
            free(errMsg);
            return rv;
        }
    }

    const char* status = "success";
    const char* messageEpilog = " data retrieved successfully";
    size_t messageEpilogStrlen = strlen(messageEpilog);
    char* message = (char*)malloc(xNameLen + messageEpilogStrlen + 1);
    if (message == NULL) {
        LOG_E("__lsapi_endpoint_x_get: Failed to allocate memory for message");
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Server ran out of memory. Poor server.");
    }
    memcpy(message, xName, xNameLen);
    memcpy(message + xNameLen, messageEpilog, messageEpilogStrlen);
    message[xNameLen + messageEpilogStrlen] = '\0';

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add x data as sub-object
    yyjson_mut_val* pXObject = yyjson_mut_obj(pJsonResp);

    switch (xType) {
        case __LSAPI_X_TYPE_REAGTYPE:
            db_reagent_type_t reagent_type = *((db_reagent_type_t*)pX);
            yyjson_mut_obj_add_int(pJsonResp, pXObject, "reagtype_id", reagent_type.reagtype_id);
            yyjson_mut_obj_add_str(pJsonResp, pXObject, "name", reagent_type.name);
            break;
        case __LSAPI_X_TYPE_REAGENT:
            db_reagent_t reagent = *((db_reagent_t*)pX);
            yyjson_mut_obj_add_int(pJsonResp, pXObject, "reagent_id", reagent.reagent_id);
            yyjson_mut_obj_add_str(pJsonResp, pXObject, "name", reagent.name);
            yyjson_mut_obj_add_str(pJsonResp, pXObject, "vendor", reagent.vendor);
            yyjson_mut_obj_add_int(pJsonResp, pXObject, "reagtype_id", reagent.reagent_type_id);
            break;
        case __LSAPI_X_TYPE_VENDOR:
            db_vendor_t vendor = *((db_vendor_t*)pX);
            yyjson_mut_obj_add_int(pJsonResp, pXObject, "vendor_id", vendor.vendor_id);
            yyjson_mut_obj_add_str(pJsonResp, pXObject, "name", vendor.name);
            break;
        case __LSAPI_X_TYPE_FACULTY:
            db_faculty_t faculty = *((db_faculty_t*)pX);
            yyjson_mut_obj_add_int(pJsonResp, pXObject, "faculty_id", faculty.faculty_id);
            yyjson_mut_obj_add_str(pJsonResp, pXObject, "name", faculty.name);
            yyjson_mut_obj_add_str(pJsonResp, pXObject, "email_domain", faculty.email_domain);
            break;
        case __LSAPI_X_TYPE_LAB:
            db_lab_t lab = *((db_lab_t*)pX);
            yyjson_mut_obj_add_int(pJsonResp, pXObject, "lab_id", lab.lab_id);
            yyjson_mut_obj_add_str(pJsonResp, pXObject, "name", lab.name);
            yyjson_mut_obj_add_int(pJsonResp, pXObject, "faculty_id", lab.faculty_id);
            break;
        case __LSAPI_X_TYPE_INVENTORY_ITEM:
            db_inventory_item_t inventory_item = *((db_inventory_item_t*)pX);
            yyjson_mut_obj_add_int(pJsonResp, pXObject, "inventory_item_id", inventory_item.inventory_id);
            yyjson_mut_obj_add_int(pJsonResp, pXObject, "reagent_id", inventory_item.reagent_id);
            yyjson_mut_obj_add_str(pJsonResp, pXObject, "date_added", inventory_item.date_added);
            yyjson_mut_obj_add_str(pJsonResp, pXObject, "date_expire", inventory_item.date_expire);
            yyjson_mut_obj_add_int(pJsonResp, pXObject, "lab_id", inventory_item.lab_id);
            yyjson_mut_obj_add_str(pJsonResp, pXObject, "epc", inventory_item.epc); // TODO remove to hide EPC from frontend?
            break;
        case __LSAPI_X_TYPE_ANTENNA:
            db_antenna_t antenna = *((db_antenna_t*)pX);
            yyjson_mut_obj_add_int(pJsonResp, pXObject, "antenna_id", antenna.antenna_id);
            yyjson_mut_obj_add_str(pJsonResp, pXObject, "name", antenna.name);
            yyjson_mut_obj_add_str(pJsonResp, pXObject, "info", antenna.info);
            yyjson_mut_obj_add_int(pJsonResp, pXObject, "k", antenna.k);
            yyjson_mut_obj_add_int(pJsonResp, pXObject, "lab_id", antenna.lab_id);
            break;
        default:
            assert(0);
    }
    // add x object to root
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, xName, pXObject);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free(xIdParamValueNt);
    free((void*)respText);
    free(message);
    yyjson_mut_doc_free(pJsonResp);

    switch (xType) {
        case __LSAPI_X_TYPE_REAGTYPE:
            db_reagent_type_free((db_reagent_type_t*)pX);
            break;
        case __LSAPI_X_TYPE_REAGENT:
            db_reagent_free((db_reagent_t*)pX);
            break;
        case __LSAPI_X_TYPE_VENDOR:
            db_vendor_free((db_vendor_t*)pX);
            break;
        case __LSAPI_X_TYPE_FACULTY:
            db_faculty_free((db_faculty_t*)pX);
            break;
        case __LSAPI_X_TYPE_LAB:
            db_lab_free((db_lab_t*)pX);
            break;
        case __LSAPI_X_TYPE_INVENTORY_ITEM:
            db_inventory_item_free((db_inventory_item_t*)pX);
            break;
        case __LSAPI_X_TYPE_ANTENNA:
            db_antenna_free((db_antenna_t*)pX);
            break;
        default:
            assert(0);
    }
    return 0;
}

/**
 * @warning You need to free the returned buffer after use
 */
static char* __lsapi_itoa(int n) {
    assert(sizeof(int) <= 4);
    char* buf = (char*)malloc(12); // 12 bytes is enough for 32-bit int
    if (buf == NULL) {
        return NULL;
    }
    snprintf(buf, 12, "%d", n);
    return buf;
}

static int __lsapi_endpoint_reagtype_get(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    return __lsapi_endpoint_x_get(pH2oHandler, pReq, pLsapi, __LSAPI_X_TYPE_REAGTYPE);
}

int lsapi_endpoint_reagtype(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);
    if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("PUT"))) {
        return __lsapi_endpoint_reagtype_put(pH2oHandler, pReq, pLsapi);
    } else if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("GET"))) {
        return __lsapi_endpoint_reagtype_get(pH2oHandler, pReq, pLsapi);
    } else {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
}

// curl -X POST -d '{"filter": "<none|name>", "value": "<value>", "p_offset": <p_offset>, "p_size": <p_size>}' http://localhost:7890/api/reagtypes
static int __lsapi_endpoint_reagtypes_post(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
    yyjson_val* pFilter = yyjson_obj_get(pRoot, "filter");
    if (pFilter == NULL || !yyjson_is_str(pFilter)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid filter");
    }
    yyjson_val* pValue = yyjson_obj_get(pRoot, "value");
    if (pValue == NULL || !yyjson_is_str(pValue)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid value");
    }
    yyjson_val* pPOffset = yyjson_obj_get(pRoot, "p_offset");
    if (pPOffset == NULL || !yyjson_is_int(pPOffset)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid p_offset");
    }
    yyjson_val* pPSize = yyjson_obj_get(pRoot, "p_size");
    if (pPSize == NULL || !yyjson_is_int(pPSize)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid p_size");
    }

    const char* filter = yyjson_get_str(pFilter);
    const char* value = yyjson_get_str(pValue);
    int pOffset = yyjson_get_int(pPOffset);
    int pSize = yyjson_get_int(pPSize);

    assert(filter != NULL && value != NULL);
    if (pOffset < 0 || pSize < 0) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "p_offset and p_size must be non-negative");
    }

    // get reagent types data from database so that we can use it for the http response
    db_reagent_type_t* reagentTypes = NULL;
    int reagentTypesCount = 0;

    char* page_offset_str = __lsapi_itoa(pOffset);
    char* page_size_str = __lsapi_itoa(pSize);
    db_reagent_type_filter_type_t filter_type = DB_REAGENT_TYPE_FILTER_TYPE_NONE;
    if (0 == strcmp(filter, "none")) {
        filter_type = DB_REAGENT_TYPE_FILTER_TYPE_NONE;
    } else if (0 == strcmp(filter, "name")) {
        filter_type = DB_REAGENT_TYPE_FILTER_TYPE_NAME;
    } else {
        yyjson_doc_free(pJson);
        free(page_offset_str);
        free(page_size_str);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid filter");
    }
    int rv = db_reagent_types_read_page_filtered(pLsapi->pDb, page_offset_str, page_size_str, &reagentTypes, &reagentTypesCount, filter_type, value);
    free(page_offset_str);
    free(page_size_str);
    if (0 != rv) {
        if (rv == -2) {
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 404, "Not Found", "No reagent types found");
        } else {
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get reagent types data from database");
        }
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Reagent types retrieved successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add reagent types data as array
    yyjson_mut_val* pReagentTypes = yyjson_mut_arr(pJsonResp);
    for (int i = 0; i < reagentTypesCount; i++) {
        yyjson_mut_val* pReagentType = yyjson_mut_obj(pJsonResp);
        yyjson_mut_obj_add_int(pJsonResp, pReagentType, "reagtype_id", reagentTypes[i].reagtype_id);
        yyjson_mut_obj_add_str(pJsonResp, pReagentType, "name", reagentTypes[i].name);
        yyjson_mut_arr_add_val(pReagentTypes, pReagentType);
    }
    // add reagent types array to root
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "reagtypes", pReagentTypes);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_doc_free(pJson);
    yyjson_mut_doc_free(pJsonResp);
    assert(reagentTypes != NULL);
    for (int i = 0; i < reagentTypesCount; i++) {
        db_reagent_type_free(&reagentTypes[i]);
    }
    free(reagentTypes);
    return 0;
}

// gets total number of reagent types in the database
static int __lsapi_endpoint_reagtypes_get(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    assert(pLsapi != NULL);
    int reagentTypesCount = 0;
    int rv = db_reagent_types_get_total_count(pLsapi->pDb, &reagentTypesCount);
    if (0 != rv) {
        LOG_E("__lsapi_endpoint_reagtypes_get: Failed to get reagent types count from database (db_reagent_types_get_total_count returned %d)", rv);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get total reagent types count from database");
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Reagent types count retrieved successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    yyjson_mut_obj_add_int(pJsonResp, pRootResp, "reagtypes_count", reagentTypesCount);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_mut_doc_free(pJsonResp);
    return 0;
}

int lsapi_endpoint_reagtypes(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);

    if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("POST"))) {
        return __lsapi_endpoint_reagtypes_post(pH2oHandler, pReq, pLsapi);
    } else if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("GET"))) {
        return __lsapi_endpoint_reagtypes_get(pH2oHandler, pReq, pLsapi);
    } else {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
}


// curl -X PUT -d '{"rname":"<reagent name>", "vendor": "<vendor>", "rtid": <reagent type id>, "username":"<username>", "session_key":"<sesskey>"}' http://localhost:7890/api/reagent
static int __lsapi_endpoint_reagent_put(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
    yyjson_val* pRName = yyjson_obj_get(pRoot, "rname");
    if (pRName == NULL || !yyjson_is_str(pRName)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rname (reagent name)");
    }
    yyjson_val* pVendor = yyjson_obj_get(pRoot, "vendor");
    if (pVendor == NULL || !yyjson_is_str(pVendor)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid vendor");
    }
    yyjson_val* pRtid = yyjson_obj_get(pRoot, "rtid");
    if (pRtid == NULL || !yyjson_is_int(pRtid)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rtid (reagent type id)");
    }
    yyjson_val* pUsername = yyjson_obj_get(pRoot, "username");
    if (pUsername == NULL || !yyjson_is_str(pUsername)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid username");
    }
    yyjson_val* pSessionKey = yyjson_obj_get(pRoot, "session_key");
    if (pSessionKey == NULL || !yyjson_is_str(pSessionKey)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid session_key in request body");
    }

    const char* rName = yyjson_get_str(pRName);
    const char* vendor = yyjson_get_str(pVendor);
    int rtid = yyjson_get_int(pRtid);
    const char* username = yyjson_get_str(pUsername);
    const char* userProvidedSessionKey = yyjson_get_str(pSessionKey);

    assert(rName != NULL && vendor != NULL && rtid >= 0 && username != NULL && userProvidedSessionKey != NULL);

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

    if (0 == strlen(user.sesskey_hash)) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Session key not set");
    }

    // verify session key
    char userProvidedSessionKeyHash[BCRYPT_HASHSIZE];
    assert(user.sesskey_hash != NULL);
    assert(strlen(user.sesskey_hash) == BCRYPT_HASHSIZE - 4);
    assert(user.sesskey_salt != NULL);
    assert(strlen(user.sesskey_salt) == (BCRYPT_HASHSIZE - 4)/2 - 1);

    assert(0 == bcrypt_hashpw(userProvidedSessionKey, user.sesskey_salt, userProvidedSessionKeyHash));
    assert(userProvidedSessionKeyHash[BCRYPT_HASHSIZE - 4] == '\0');
    assert(strlen(userProvidedSessionKeyHash) == BCRYPT_HASHSIZE - 4);
    LOG_V("__lsapi_endpoint_reagent_put: user-provided session key: %s", userProvidedSessionKey);
    LOG_V("__lsapi_endpoint_reagent_put: userProvidedSessionKeyHash: %s, user.sesskey_hash: %s, user.sesskey_salt: %s", userProvidedSessionKeyHash, user.sesskey_hash, user.sesskey_salt);

    if (0 != strcmp(userProvidedSessionKeyHash, user.sesskey_hash)) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Invalid session key");
    }

    // create reagent + get reagent data from database so that we can use it for the http response
    db_reagent_t reagent;
    char* rtid_str = __lsapi_itoa(rtid);
    if (0 != db_reagent_insert_ret(pLsapi->pDb, rName, vendor, rtid_str, &reagent)) {
        yyjson_doc_free(pJson);
        free(rtid_str);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to create reagent");
    }
    free(rtid_str);

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Reagent created successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add reagent data as sub-object
    yyjson_mut_val* pReagent = yyjson_mut_obj(pJsonResp);
    yyjson_mut_obj_add_int(pJsonResp, pReagent, "reagent_id", reagent.reagent_id);
    yyjson_mut_obj_add_str(pJsonResp, pReagent, "name", reagent.name);
    yyjson_mut_obj_add_str(pJsonResp, pReagent, "vendor", reagent.vendor);
    yyjson_mut_obj_add_int(pJsonResp, pReagent, "reagtype_id", reagent.reagent_type_id);
    // add reagent object to root
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "reagent", pReagent);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_doc_free(pJson);
    yyjson_mut_doc_free(pJsonResp);
    db_user_free(&user);
    db_reagent_free(&reagent);
    return 0;
}

static int __lsapi_endpoint_reagent_get(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    assert(pLsapi != NULL);
    if (pReq->query_at == SIZE_MAX) {
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing query string");
    }
    char* queryStr = pReq->path.base + pReq->query_at + 1;
    size_t queryStrLen = pReq->path.len - pReq->query_at - 1;
    LOG_D("__lsapi_endpoint_reagent_get: queryStrLen = %lu", queryStrLen);
    LOG_D("__lsapi_endpoint_reagent_get: queryStr = %.*s", (int)queryStrLen, queryStr);

    const char* reagentIdParamName = "reagent_id";
    size_t reagentIdParamNameLen = strlen(reagentIdParamName);
    if (queryStrLen < 1) {
        LOG_D("__lsapi_endpoint_reagent_get: Empty query string (queryStrLen = %lu)", queryStrLen);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Empty query string");
    } else if (queryStrLen < reagentIdParamNameLen + 2) { // reagent_id=d is the shortest possible query string (where d is a decimal digit)
        LOG_D("__lsapi_endpoint_reagent_get: Query string too short (queryStrLen = %lu)", queryStrLen);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Query string too short");
    }
    char* reagentIdParamNameAddr = strstr(queryStr, reagentIdParamName);
    if (reagentIdParamNameAddr == NULL) {
        LOG_D("__lsapi_endpoint_reagent_get: Missing reagent_id parameter in query string");
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing reagent_id parameter in query string");
    } else if (reagentIdParamNameAddr != queryStr) {
        LOG_D("__lsapi_endpoint_reagent_get: reagent_id parameter not at the beginning of query string");
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "reagent_id parameter not at the beginning of query string");
    }
    char* reagentIdParamNVSeparatorAddr = reagentIdParamNameAddr + reagentIdParamNameLen;
    if (*reagentIdParamNVSeparatorAddr != '=') {
        LOG_D("__lsapi_endpoint_reagent_get: Missing = after reagent_id in query string");
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing = after reagent_id in query string");
    }
    char* reagentIdParamValue = reagentIdParamNVSeparatorAddr + 1;
    size_t reagentIdParamValueLen = queryStr + queryStrLen - reagentIdParamValue;
    assert(reagentIdParamValueLen >= 1);
    for (size_t i = 0; i < reagentIdParamValueLen; i++) {
        if (!isdigit(reagentIdParamValue[i])) {
            LOG_D("__lsapi_endpoint_reagent_get: Invalid reagent_id value in query string (non-digit character at position %lu in string %.*s)", i, (int)reagentIdParamValueLen, reagentIdParamValue);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid reagent_id value in query string");
        }
    }
    char* reagentIdParamValueNt = (char*)malloc(reagentIdParamValueLen + 1);
    if (reagentIdParamValueNt == NULL) {
        LOG_E("__lsapi_endpoint_reagent_get: Failed to allocate memory for reagentIdParamValueNt");
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Server ran out of memory. Poor server.");
    }
    memcpy(reagentIdParamValueNt, reagentIdParamValue, reagentIdParamValueLen);
    reagentIdParamValueNt[reagentIdParamValueLen] = '\0';
    int reagentId = atoi(reagentIdParamValueNt);
    LOG_V("__lsapi_endpoint_reagent_get: reagentId = %d", reagentId);
    assert(reagentId >= 0);

    // get reagent data from database so that we can use it for the http response
    db_reagent_t reagent;
    int rv = db_reagent_get_by_id(pLsapi->pDb, reagentIdParamValueNt, &reagent);
    if (0 != rv) {
        if (rv == -2) {
            return __lsapi_endpoint_error(pReq, 404, "Not Found", "Reagent not found");
        } else {
            LOG_E("__lsapi_endpoint_reagent_get: Failed to get reagent data from database (db_reagent_get_by_id returned %d)", rv);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get reagent data from database");
        }
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Reagent data retrieved successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add reagent data as sub-object
    yyjson_mut_val* pReagent = yyjson_mut_obj(pJsonResp);
    yyjson_mut_obj_add_int(pJsonResp, pReagent, "reagent_id", reagent.reagent_id);
    yyjson_mut_obj_add_str(pJsonResp, pReagent, "name", reagent.name);
    yyjson_mut_obj_add_str(pJsonResp, pReagent, "vendor", reagent.vendor);
    yyjson_mut_obj_add_int(pJsonResp, pReagent, "reagtype_id", reagent.reagent_type_id);
    // add reagent object to root
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "reagent", pReagent);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free(reagentIdParamValueNt);
    free((void*)respText);
    yyjson_mut_doc_free(pJsonResp);
    db_reagent_free(&reagent);
    return 0;
}

int lsapi_endpoint_reagent(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);
    if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("PUT"))) {
        return __lsapi_endpoint_reagent_put(pH2oHandler, pReq, pLsapi);
    } else if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("GET"))) {
        return __lsapi_endpoint_reagent_get(pH2oHandler, pReq, pLsapi); 
    }
    else {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
}

// Obtain list of reagents
//TODO Authentication & authorization?
// curl -X POST -d '{"filter": "<none|name|vendor|reagtype_id>", "value": "<value>", "p_offset": <p_offset>, "p_size": <p_size>}' http://localhost:7890/api/reagents
static int __lsapi_endpoint_reagents_post(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
    yyjson_val* pFilter = yyjson_obj_get(pRoot, "filter");
    if (pFilter == NULL || !yyjson_is_str(pFilter)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid filter");
    }
    yyjson_val* pValue = yyjson_obj_get(pRoot, "value");
    if (pValue == NULL || !yyjson_is_str(pValue)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid value");
    }
    yyjson_val* pPOffset = yyjson_obj_get(pRoot, "p_offset");
    if (pPOffset == NULL || !yyjson_is_int(pPOffset)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid p_offset");
    }
    yyjson_val* pPSize = yyjson_obj_get(pRoot, "p_size");
    if (pPSize == NULL || !yyjson_is_int(pPSize)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid p_size");
    }
    
    const char* filter = yyjson_get_str(pFilter);
    const char* value = yyjson_get_str(pValue);
    int page_offset = yyjson_get_int(pPOffset);
    int page_size = yyjson_get_int(pPSize);

    assert(filter != NULL && value != NULL);
    if (page_offset < 0 || page_size < 1) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid p_offset or p_size");
    }

    // get reagents data from database so that we can use it for the http response
    db_reagent_t* reagents = NULL;
    int reagents_count = 0;

    char* page_offset_str = __lsapi_itoa(page_offset);
    char* page_size_str = __lsapi_itoa(page_size);
    db_reagent_filter_type_t filter_type = DB_REAGENT_FILTER_TYPE_NONE;
    if (0 == strcmp(filter, "none")) {
        filter_type = DB_REAGENT_FILTER_TYPE_NONE;
    } else if (0 == strcmp(filter, "name")) {
        filter_type = DB_REAGENT_FILTER_TYPE_NAME;
    } else if (0 == strcmp(filter, "vendor")) {
        filter_type = DB_REAGENT_FILTER_TYPE_VENDOR;
    } else if (0 == strcmp(filter, "reagtype_id")) {
        filter_type = DB_REAGENT_FILTER_TYPE_REAGTYPE_ID;
    } else if (0 == strcmp(filter, "reagtype_name")) {
        filter_type = DB_REAGENT_FILTER_TYPE_REAGTYPE_NAME;
    } else {
        yyjson_doc_free(pJson);
        free(page_offset_str);
        free(page_size_str);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid filter");
    }
    int rv = db_reagents_read_page_filtered(pLsapi->pDb, page_offset_str, page_size_str, &reagents, &reagents_count, filter_type, value);
    free(page_offset_str);
    free(page_size_str);
    if (0 != rv) {
        if (rv == -2) {
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 404, "Not Found", "No reagents found");
        } else {
            LOG_E("__lsapi_endpoint_reagents_post: Failed to get reagents data from database (db_reagents_read_page_filtered returned %d)", rv);
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get reagents data from database");
        }
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Reagents data retrieved successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add reagents data as array
    yyjson_mut_val* pReagents = yyjson_mut_arr(pJsonResp);
    for (int i = 0; i < reagents_count; i++) {
        yyjson_mut_val* pReagent = yyjson_mut_obj(pJsonResp);
        yyjson_mut_obj_add_int(pJsonResp, pReagent, "reagent_id", reagents[i].reagent_id);
        yyjson_mut_obj_add_str(pJsonResp, pReagent, "name", reagents[i].name);
        yyjson_mut_obj_add_str(pJsonResp, pReagent, "vendor", reagents[i].vendor);
        yyjson_mut_obj_add_int(pJsonResp, pReagent, "reagtype_id", reagents[i].reagent_type_id);
        yyjson_mut_arr_add_val(pReagents, pReagent);
    }
    // add reagents array to root
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "reagents", pReagents);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_doc_free(pJson);
    yyjson_mut_doc_free(pJsonResp);
    assert(reagents != NULL);
    for (int i = 0; i < reagents_count; i++) {
        db_reagent_free(&reagents[i]);
    }
    free(reagents);
    return 0;
}

// gets total number of reagents in the database
static int __lsapi_endpoint_reagents_get(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    assert(pLsapi != NULL);
    int reagents_count = 0;
    int rv = db_reagents_get_total_count(pLsapi->pDb, &reagents_count);
    if (0 != rv) {
        LOG_E("__lsapi_endpoint_reagents_get: Failed to get total reagents count from database (db_reagents_get_total_count returned %d)", rv);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get total reagents count from database");
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Reagents count retrieved successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    yyjson_mut_obj_add_int(pJsonResp, pRootResp, "reagents_count", reagents_count);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);
    
    free((void*)respText);
    yyjson_mut_doc_free(pJsonResp);
    return 0;
}

int lsapi_endpoint_reagents(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);
    if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("POST"))) {
        return __lsapi_endpoint_reagents_post(pH2oHandler, pReq, pLsapi);
    } else if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("GET"))) {
        return __lsapi_endpoint_reagents_get(pH2oHandler, pReq, pLsapi);
    } else {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
}

static int __lsapi_endpoint_vendor_get(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    return __lsapi_endpoint_x_get(pH2oHandler, pReq, pLsapi, __LSAPI_X_TYPE_VENDOR);
}

int lsapi_endpoint_vendor(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);
    if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("GET"))) {
        return __lsapi_endpoint_vendor_get(pH2oHandler, pReq, pLsapi);
    } else {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
}

//TODO Abstract these into a common function (__lsapi_endpoint_xs_post maybe?)
// curl -X POST -d '{"filter": "<none|name>", "value": "<value>", "p_offset": <p_offset>, "p_size": <p_size>}' http://localhost:7890/api/vendors
static int __lsapi_endpoint_vendors_post(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
    yyjson_val* pFilter = yyjson_obj_get(pRoot, "filter");
    if (pFilter == NULL || !yyjson_is_str(pFilter)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid filter");
    }
    yyjson_val* pValue = yyjson_obj_get(pRoot, "value");
    if (pValue == NULL || !yyjson_is_str(pValue)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid value");
    }
    yyjson_val* pPOffset = yyjson_obj_get(pRoot, "p_offset");
    if (pPOffset == NULL || !yyjson_is_int(pPOffset)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid p_offset");
    }
    yyjson_val* pPSize = yyjson_obj_get(pRoot, "p_size");
    if (pPSize == NULL || !yyjson_is_int(pPSize)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid p_size");
    }

    const char* filter = yyjson_get_str(pFilter);
    const char* value = yyjson_get_str(pValue);
    int page_offset = yyjson_get_int(pPOffset);
    int page_size = yyjson_get_int(pPSize);

    assert(filter != NULL && value != NULL);
    if (page_offset < 0 || page_size < 1) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid p_offset or p_size");
    }

    // get vendors data from database so that we can use it for the http response
    db_vendor_t* vendors = NULL;
    int vendors_count = 0;

    char* page_offset_str = __lsapi_itoa(page_offset);
    char* page_size_str = __lsapi_itoa(page_size);
    db_vendor_filter_type_t filter_type = DB_VENDOR_FILTER_TYPE_NONE;
    if (0 == strcmp(filter, "none")) {
        filter_type = DB_VENDOR_FILTER_TYPE_NONE;
    } else if (0 == strcmp(filter, "name")) {
        filter_type = DB_VENDOR_FILTER_TYPE_NAME;
    } else {
        yyjson_doc_free(pJson);
        free(page_offset_str);
        free(page_size_str);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid filter");
    }
    int rv = db_vendors_read_page_filtered(pLsapi->pDb, page_offset_str, page_size_str, &vendors, &vendors_count, filter_type, value);
    free(page_offset_str);
    free(page_size_str);
    if (0 != rv) {
        if (rv == -2) {
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 404, "Not Found", "No vendors found");
        } else {
            LOG_E("__lsapi_endpoint_vendors_post: Failed to get vendors data from database (db_vendors_read_page_filtered returned %d)", rv);
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get vendors data from database");
        }
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Vendors data retrieved successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add vendors data as array
    yyjson_mut_val* pVendors = yyjson_mut_arr(pJsonResp);
    for (int i = 0; i < vendors_count; i++) {
        yyjson_mut_val* pVendor = yyjson_mut_obj(pJsonResp);
        yyjson_mut_obj_add_int(pJsonResp, pVendor, "vendor_id", vendors[i].vendor_id);
        yyjson_mut_obj_add_str(pJsonResp, pVendor, "name", vendors[i].name);
        yyjson_mut_arr_add_val(pVendors, pVendor);
    }
    // add vendors array to root
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "vendors", pVendors);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_doc_free(pJson);
    yyjson_mut_doc_free(pJsonResp);
    assert(vendors != NULL);
    for (int i = 0; i < vendors_count; i++) {
        db_vendor_free(&vendors[i]);
    }
    free(vendors);
    return 0;
}

// gets total number of vendors in the database
static int __lsapi_endpoint_vendors_get(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    assert(pLsapi != NULL);
    int vendors_count = 0;
    int rv = db_vendors_get_total_count(pLsapi->pDb, &vendors_count);
    if (0 != rv) {
        LOG_E("__lsapi_endpoint_vendors_get: Failed to get total vendors count from database (db_vendors_get_total_count returned %d)", rv);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get total vendors count from database");
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Vendors count retrieved successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    yyjson_mut_obj_add_int(pJsonResp, pRootResp, "vendors_count", vendors_count);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);
    
    free((void*)respText);
    yyjson_mut_doc_free(pJsonResp);
    return 0;
}

int lsapi_endpoint_vendors(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);
    if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("POST"))) {
        return __lsapi_endpoint_vendors_post(pH2oHandler, pReq, pLsapi);
    } else if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("GET"))) {
        return __lsapi_endpoint_vendors_get(pH2oHandler, pReq, pLsapi);
    } else {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
}

// curl -X PUT -d '{"fname":"<faculty name>", "fedomain":"<faculty email domain>", "username":"<username>", "session_key":"<sesskey>"}'
static int __lsapi_endpoint_faculty_put(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
    yyjson_val* pFName = yyjson_obj_get(pRoot, "fname");
    if (pFName == NULL || !yyjson_is_str(pFName)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid fname (faculty name)");
    }
    yyjson_val* pFeDomain = yyjson_obj_get(pRoot, "fedomain");
    if (pFeDomain == NULL || !yyjson_is_str(pFeDomain)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid fedomain (faculty email domain)");
    }
    yyjson_val* pUsername = yyjson_obj_get(pRoot, "username");
    if (pUsername == NULL || !yyjson_is_str(pUsername)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid username");
    }
    yyjson_val* pSessionKey = yyjson_obj_get(pRoot, "session_key");
    if (pSessionKey == NULL || !yyjson_is_str(pSessionKey)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid session_key in request body");
    }

    const char* fName = yyjson_get_str(pFName);
    const char* feDomain = yyjson_get_str(pFeDomain);
    const char* username = yyjson_get_str(pUsername);
    const char* userProvidedSessionKey = yyjson_get_str(pSessionKey);

    assert(fName != NULL && feDomain != NULL && username != NULL && userProvidedSessionKey != NULL);

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

    if (0 == strlen(user.sesskey_hash)) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Session key not set");
    }

    // verify session key
    char userProvidedSessionKeyHash[BCRYPT_HASHSIZE];
    assert(user.sesskey_hash != NULL);
    assert(strlen(user.sesskey_hash) == BCRYPT_HASHSIZE - 4);
    assert(user.sesskey_salt != NULL);
    assert(strlen(user.sesskey_salt) == (BCRYPT_HASHSIZE - 4)/2 - 1);

    assert(0 == bcrypt_hashpw(userProvidedSessionKey, user.sesskey_salt, userProvidedSessionKeyHash));
    assert(userProvidedSessionKeyHash[BCRYPT_HASHSIZE - 4] == '\0');
    assert(strlen(userProvidedSessionKeyHash) == BCRYPT_HASHSIZE - 4);
    LOG_V("__lsapi_endpoint_faculty_put: user-provided session key: %s", userProvidedSessionKey);
    LOG_V("__lsapi_endpoint_faculty_put: userProvidedSessionKeyHash: %s, user.sesskey_hash: %s, user.sesskey_salt: %s", userProvidedSessionKeyHash, user.sesskey_hash, user.sesskey_salt);

    if (0 != strcmp(userProvidedSessionKeyHash, user.sesskey_hash)) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Invalid session key");
    }

    // create faculty + get faculty data from database so that we can use it for the http response
    db_faculty_t faculty;
    if (0 != db_faculty_insert_ret(pLsapi->pDb, fName, feDomain, &faculty)) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to create faculty");
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Faculty created successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add faculty data as sub-object
    yyjson_mut_val* pFaculty = yyjson_mut_obj(pJsonResp);
    yyjson_mut_obj_add_int(pJsonResp, pFaculty, "faculty_id", faculty.faculty_id);
    yyjson_mut_obj_add_str(pJsonResp, pFaculty, "name", faculty.name);
    yyjson_mut_obj_add_str(pJsonResp, pFaculty, "email_domain", faculty.email_domain);
    // add faculty object to root
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "faculty", pFaculty);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_doc_free(pJson);
    yyjson_mut_doc_free(pJsonResp);
    db_user_free(&user);
    db_faculty_free(&faculty);
    return 0;
}

static int __lsapi_endpoint_faculty_get(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    return __lsapi_endpoint_x_get(pH2oHandler, pReq, pLsapi, __LSAPI_X_TYPE_FACULTY);
}

int lsapi_endpoint_faculty(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);
    if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("PUT"))) {
        return __lsapi_endpoint_faculty_put(pH2oHandler, pReq, pLsapi);
    } else if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("GET"))) {
        return __lsapi_endpoint_faculty_get(pH2oHandler, pReq, pLsapi);
    } else {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
}

// curl -X POST -d '{"filter": "<none|name|uid|email_domain>", "value": "<value>", "p_offset": <p_offset>, "p_size": <p_size>}' http://localhost:7890/api/faculties
static int __lsapi_endpoint_faculties_post(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
    yyjson_val* pFilter = yyjson_obj_get(pRoot, "filter");
    if (pFilter == NULL || !yyjson_is_str(pFilter)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid filter");
    }
    yyjson_val* pValue = yyjson_obj_get(pRoot, "value");
    if (pValue == NULL || !yyjson_is_str(pValue)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid value");
    }
    yyjson_val* pPOffset = yyjson_obj_get(pRoot, "p_offset");
    if (pPOffset == NULL || !yyjson_is_int(pPOffset)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid p_offset");
    }
    yyjson_val* pPSize = yyjson_obj_get(pRoot, "p_size");
    if (pPSize == NULL || !yyjson_is_int(pPSize)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid p_size");
    }

    const char* filter = yyjson_get_str(pFilter);
    const char* value = yyjson_get_str(pValue);
    int page_offset = yyjson_get_int(pPOffset);
    int page_size = yyjson_get_int(pPSize);

    assert(filter != NULL && value != NULL);
    if (page_offset < 0 || page_size < 1) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid p_offset or p_size");
    }

    // get faculties data from database so that we can use it for the http response
    db_faculty_t* faculties = NULL;
    int faculties_count = 0;

    char* page_offset_str = __lsapi_itoa(page_offset);
    char* page_size_str = __lsapi_itoa(page_size);
    db_faculty_filter_type_t filter_type = DB_FACULTY_FILTER_TYPE_NONE;
    if (0 == strcmp(filter, "none")) {
        filter_type = DB_FACULTY_FILTER_TYPE_NONE;
    } else if (0 == strcmp(filter, "name")) {
        filter_type = DB_FACULTY_FILTER_TYPE_NAME;
    } else if (0 == strcmp(filter, "uid")) {
        filter_type = DB_FACULTY_FILTER_TYPE_USER_ID;
    } else if (0 == strcmp(filter, "email_domain")) {
        filter_type = DB_FACULTY_FILTER_TYPE_EMAIL_DOMAIN;
    } else {
        yyjson_doc_free(pJson);
        free(page_offset_str);
        free(page_size_str);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid filter");
    }
    int rv = db_faculties_read_page_filtered(pLsapi->pDb, page_offset_str, page_size_str, &faculties, &faculties_count, filter_type, value);
    free(page_offset_str);
    free(page_size_str);
    if (0 != rv) {
        if (rv == -2) {
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 404, "Not Found", "No faculties found");
        } else {
            LOG_E("__lsapi_endpoint_faculties_post: Failed to get faculties data from database (db_faculties_read_page_filtered returned %d)", rv);
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get faculties data from database");
        }
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Faculties data retrieved successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add faculties data as array
    yyjson_mut_val* pFaculties = yyjson_mut_arr(pJsonResp);
    for (int i = 0; i < faculties_count; i++) {
        yyjson_mut_val* pFaculty = yyjson_mut_obj(pJsonResp);
        yyjson_mut_obj_add_int(pJsonResp, pFaculty, "faculty_id", faculties[i].faculty_id);
        yyjson_mut_obj_add_str(pJsonResp, pFaculty, "name", faculties[i].name);
        yyjson_mut_obj_add_str(pJsonResp, pFaculty, "email_domain", faculties[i].email_domain);
        yyjson_mut_arr_add_val(pFaculties, pFaculty);
    }
    // add faculties array to root
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "faculties", pFaculties);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_doc_free(pJson);
    yyjson_mut_doc_free(pJsonResp);
    assert(faculties != NULL);
    for (int i = 0; i < faculties_count; i++) {
        db_faculty_free(&faculties[i]);
    }
    free(faculties);
    return 0;
}

//gets total number of faculties in the database
static int __lsapi_endpoint_faculties_get(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    assert(pLsapi != NULL);
    int faculties_count = 0;
    int rv = db_faculties_get_total_count(pLsapi->pDb, &faculties_count);
    if (0 != rv) {
        LOG_E("__lsapi_endpoint_faculties_get: Failed to get total faculties count from database (db_faculties_get_total_count returned %d)", rv);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get total faculties count from database");
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Faculties count retrieved successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    yyjson_mut_obj_add_int(pJsonResp, pRootResp, "faculties_count", faculties_count);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);
    
    free((void*)respText);
    yyjson_mut_doc_free(pJsonResp);
    return 0;
}

int lsapi_endpoint_faculties(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);
    if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("POST"))) {
        return __lsapi_endpoint_faculties_post(pH2oHandler, pReq, pLsapi);
    } else if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("GET"))) {
        return __lsapi_endpoint_faculties_get(pH2oHandler, pReq, pLsapi);
    } else {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
}

// curl -X PUT -d '{"lname":"<lab name>", "ltoken":"<lab token>", "lkey": "<lkey>", "host": "<host>", "fid":<faculty id>, "username":"<username>", "session_key":"<sesskey>"}'
static int __lsapi_endpoint_lab_put(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
    yyjson_val* pLName = yyjson_obj_get(pRoot, "lname");
    if (pLName == NULL || !yyjson_is_str(pLName)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid lname (lab name)");
    }
    yyjson_val* pLToken = yyjson_obj_get(pRoot, "ltoken");
    if (pLToken == NULL || !yyjson_is_str(pLToken)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid ltoken (lab token)");
    }
    yyjson_val* pLkey = yyjson_obj_get(pRoot, "lkey");
    if (pLkey == NULL || !yyjson_is_str(pLkey)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid lkey");
    }
    yyjson_val* pHost = yyjson_obj_get(pRoot, "host");
    if (pHost == NULL || !yyjson_is_str(pHost)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid host");
    }
    yyjson_val* pFid = yyjson_obj_get(pRoot, "fid");
    if (pFid == NULL || !yyjson_is_int(pFid)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid fid (faculty id)");
    }
    yyjson_val* pUsername = yyjson_obj_get(pRoot, "username");
    if (pUsername == NULL || !yyjson_is_str(pUsername)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid username");
    }
    yyjson_val* pSessionKey = yyjson_obj_get(pRoot, "session_key");
    if (pSessionKey == NULL || !yyjson_is_str(pSessionKey)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid session_key in request body");
    }

    const char* lName = yyjson_get_str(pLName);
    const char* lToken = yyjson_get_str(pLToken);
    const char* lKey = yyjson_get_str(pLkey);
    const char* host = yyjson_get_str(pHost);
    int fid = yyjson_get_int(pFid);
    const char* username = yyjson_get_str(pUsername);
    const char* userProvidedSessionKey = yyjson_get_str(pSessionKey);

    assert(lName != NULL && lToken != NULL && lKey != NULL && host != NULL && fid >= 0 && username != NULL && userProvidedSessionKey != NULL);

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

    if (0 == strlen(user.sesskey_hash)) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Session key not set");
    }

    // verify session key
    char userProvidedSessionKeyHash[BCRYPT_HASHSIZE];
    assert(user.sesskey_hash != NULL);
    assert(strlen(user.sesskey_hash) == BCRYPT_HASHSIZE - 4);
    assert(user.sesskey_salt != NULL);
    assert(strlen(user.sesskey_salt) == (BCRYPT_HASHSIZE - 4)/2 - 1);

    assert(0 == bcrypt_hashpw(userProvidedSessionKey, user.sesskey_salt, userProvidedSessionKeyHash));
    assert(userProvidedSessionKeyHash[BCRYPT_HASHSIZE - 4] == '\0');
    assert(strlen(userProvidedSessionKeyHash) == BCRYPT_HASHSIZE - 4);
    LOG_V("__lsapi_endpoint_lab_put: user-provided session key: %s", userProvidedSessionKey);
    LOG_V("__lsapi_endpoint_lab_put: userProvidedSessionKeyHash: %s, user.sesskey_hash: %s, user.sesskey_salt: %s", userProvidedSessionKeyHash, user.sesskey_hash, user.sesskey_salt);

    if (0 != strcmp(userProvidedSessionKeyHash, user.sesskey_hash)) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Invalid session key");
    }

    // hash ltoken
    char lTokenHash[BCRYPT_HASHSIZE];
    char lTokenSalt[BCRYPT_HASHSIZE];
    assert(0 == bcrypt_gensalt(12, lTokenSalt));
    assert(0 == bcrypt_hashpw(lToken, lTokenSalt, lTokenHash));
    assert(lTokenHash[BCRYPT_HASHSIZE - 4] == '\0');
    assert(strlen(lTokenHash) == BCRYPT_HASHSIZE - 4);
    assert(lTokenSalt[BCRYPT_HASHSIZE/2 - 1] == '\0');
    assert(strlen(lTokenSalt) == (BCRYPT_HASHSIZE - 4)/2 - 1);

    // create lab + get lab data from database so that we can use it for the http response
    db_lab_t lab;
    char* fid_str = __lsapi_itoa(fid);
    if (0 != db_lab_insert_ret(pLsapi->pDb, lName, lTokenHash, lTokenSalt, lKey, host, fid_str, &lab)) {
        yyjson_doc_free(pJson);
        free(fid_str);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to create lab");
    }
    free(fid_str);

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Lab created successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add lab public data as sub-object
    yyjson_mut_val* pLab = yyjson_mut_obj(pJsonResp);
    yyjson_mut_obj_add_int(pJsonResp, pLab, "lab_id", lab.lab_id);
    yyjson_mut_obj_add_str(pJsonResp, pLab, "name", lab.name);
    yyjson_mut_obj_add_str(pJsonResp, pLab, "host", lab.host);
    yyjson_mut_obj_add_int(pJsonResp, pLab, "faculty_id", lab.faculty_id);
    // add lab object to root
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "lab", pLab);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_doc_free(pJson);
    yyjson_mut_doc_free(pJsonResp);
    db_user_free(&user);
    db_lab_free(&lab);
    return 0;
}

static int __lsapi_endpoint_lab_get(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    return __lsapi_endpoint_x_get(pH2oHandler, pReq, pLsapi, __LSAPI_X_TYPE_LAB);
}

int lsapi_endpoint_lab(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);
    if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("PUT"))) {
        return __lsapi_endpoint_lab_put(pH2oHandler, pReq, pLsapi);
    } else if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("GET"))) {
        return __lsapi_endpoint_x_get(pH2oHandler, pReq, pLsapi, __LSAPI_X_TYPE_LAB);
    } else {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    } 
}

// curl -X POST -d {"filter": "<none|name|fid|uid>", "value": "<value>", "p_offset": <p_offset>, "p_size": <p_size>}' http://localhost:7890/api/labs
static int __lsapi_endpoint_labs_post(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
    yyjson_val* pFilter = yyjson_obj_get(pRoot, "filter");
    if (pFilter == NULL || !yyjson_is_str(pFilter)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid filter");
    }
    yyjson_val* pValue = yyjson_obj_get(pRoot, "value");
    if (pValue == NULL || !yyjson_is_str(pValue)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid value");
    }
    yyjson_val* pPOffset = yyjson_obj_get(pRoot, "p_offset");
    if (pPOffset == NULL || !yyjson_is_int(pPOffset)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid p_offset");
    }
    yyjson_val* pPSize = yyjson_obj_get(pRoot, "p_size");
    if (pPSize == NULL || !yyjson_is_int(pPSize)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid p_size");
    }

    const char* filter = yyjson_get_str(pFilter);
    const char* value = yyjson_get_str(pValue);
    int page_offset = yyjson_get_int(pPOffset);
    int page_size = yyjson_get_int(pPSize);

    assert(filter != NULL && value != NULL);
    if (page_offset < 0 || page_size < 0) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid p_offset or p_size");
    }

    // get labs data from database so that we can use it for the http response
    db_lab_t* labs = NULL;
    int labs_count = 0;

    char* page_offset_str = __lsapi_itoa(page_offset);
    char* page_size_str = __lsapi_itoa(page_size);
    db_lab_filter_type_t filter_type = DB_LAB_FILTER_TYPE_NONE;
    if (0 == strcmp(filter, "none")) {
        filter_type = DB_LAB_FILTER_TYPE_NONE;
    } else if (0 == strcmp(filter, "name")) {
        filter_type = DB_LAB_FILTER_TYPE_NAME;
    } else if (0 == strcmp(filter, "fid")) {
        filter_type = DB_LAB_FILTER_TYPE_FACULTY_ID;
    } else if (0 == strcmp(filter, "uid")) {
        filter_type = DB_LAB_FILTER_TYPE_USER_ID;
    } else {
        yyjson_doc_free(pJson);
        free(page_offset_str);
        free(page_size_str);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid filter");
    }
    int rv = db_labs_read_page_filtered(pLsapi->pDb, page_offset_str, page_size_str, &labs, &labs_count, filter_type, value);
    free(page_offset_str);
    free(page_size_str);
    if (0 != rv) {
        if (rv == -2) {
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 404, "Not Found", "No labs found");
        } else {
            LOG_E("__lsapi_endpoint_labs_post: Failed to get labs data from database (db_labs_read_page_filtered returned %d)", rv);
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get labs data from database");
        }
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Labs data retrieved successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add labs data as array
    yyjson_mut_val* pLabs = yyjson_mut_arr(pJsonResp);
    for (int i = 0; i < labs_count; i++) {
        yyjson_mut_val* pLab = yyjson_mut_obj(pJsonResp);
        yyjson_mut_obj_add_int(pJsonResp, pLab, "lab_id", labs[i].lab_id);
        yyjson_mut_obj_add_str(pJsonResp, pLab, "name", labs[i].name);
        yyjson_mut_obj_add_int(pJsonResp, pLab, "faculty_id", labs[i].faculty_id);
        yyjson_mut_arr_add_val(pLabs, pLab);
    }
    // add labs array to root
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "labs", pLabs);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_doc_free(pJson);
    yyjson_mut_doc_free(pJsonResp);
    assert(labs != NULL);
    for (int i = 0; i < labs_count; i++) {
        db_lab_free(&labs[i]);
    }
    free(labs);
    return 0;
}

// gets total number of labs in the database
static int __lsapi_endpoint_labs_get(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    assert(pLsapi != NULL);
    int labsCount = 0;
    int rv = db_labs_get_total_count(pLsapi->pDb, &labsCount);
    if (0 != rv) {
        LOG_E("__lsapi_endpoint_labs_get: Failed to get total labs count from database (db_labs_get_total_count returned %d)", rv);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get total labs count from database");
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Labs count retrieved successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    yyjson_mut_obj_add_int(pJsonResp, pRootResp, "labs_count", labsCount);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_mut_doc_free(pJsonResp);
    return 0;
}

int lsapi_endpoint_labs(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);

    if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("POST"))) {
        return __lsapi_endpoint_labs_post(pH2oHandler, pReq, pLsapi);
    } else if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("GET"))) {
        return __lsapi_endpoint_labs_get(pH2oHandler, pReq, pLsapi);
    } else {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
}

/**
 * TODO should we really pass epc, apwd, kpwd instead of generate it? For now yes, as it seems to make debugging easier
*/
// curl -X PUT -d '{"rgid":<reagent id>, "dadd":"<date added>", "dexp": "<date expire>", "lid":<lab id>, "epc": "<epc>", "apwd": "<apwd>", "kpwd":"<kpwd>", "username":"<username>", "session_key":"<sesskey>"}'
static int __lsapi_endpoint_inventory_put(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
    yyjson_val* pRgid = yyjson_obj_get(pRoot, "rgid");
    if (pRgid == NULL || !yyjson_is_int(pRgid)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rgid (reagent id)");
    }
    yyjson_val* pDadd = yyjson_obj_get(pRoot, "dadd");
    if (pDadd == NULL || !yyjson_is_str(pDadd)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid dadd (date added)");
    }
    yyjson_val* pDexp = yyjson_obj_get(pRoot, "dexp");
    if (pDexp == NULL || !yyjson_is_str(pDexp)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid dexp (date expire)");
    }
    yyjson_val* pLid = yyjson_obj_get(pRoot, "lid");
    if (pLid == NULL || !yyjson_is_int(pLid)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid lid (lab id)");
    }
    yyjson_val* pEpc = yyjson_obj_get(pRoot, "epc");
    if (pEpc == NULL || !yyjson_is_str(pEpc)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid epc");
    }
    yyjson_val* pApwd = yyjson_obj_get(pRoot, "apwd");
    if (pApwd == NULL || !yyjson_is_str(pApwd)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid apwd");
    }
    yyjson_val* pKpwd = yyjson_obj_get(pRoot, "kpwd");
    if (pKpwd == NULL || !yyjson_is_str(pKpwd)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid kpwd");
    }
    yyjson_val* pUsername = yyjson_obj_get(pRoot, "username");
    if (pUsername == NULL || !yyjson_is_str(pUsername)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid username");
    }
    yyjson_val* pSessionKey = yyjson_obj_get(pRoot, "session_key");
    if (pSessionKey == NULL || !yyjson_is_str(pSessionKey)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid session_key in request body");
    }

    int rgid = yyjson_get_int(pRgid);
    const char* dadd = yyjson_get_str(pDadd);
    const char* dexp = yyjson_get_str(pDexp);
    int lid = yyjson_get_int(pLid);
    const char* epc = yyjson_get_str(pEpc);
    const char* apwd = yyjson_get_str(pApwd);
    const char* kpwd = yyjson_get_str(pKpwd);
    const char* username = yyjson_get_str(pUsername);
    const char* userProvidedSessionKey = yyjson_get_str(pSessionKey);

    assert(rgid >= 0 && dadd != NULL && dexp != NULL && lid >= 0 && epc != NULL && apwd != NULL && kpwd != NULL && username != NULL && userProvidedSessionKey != NULL);

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

    if (0 == strlen(user.sesskey_hash)) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Session key not set");
    }

    // verify session key
    char userProvidedSessionKeyHash[BCRYPT_HASHSIZE];
    assert(user.sesskey_hash != NULL);
    assert(strlen(user.sesskey_hash) == BCRYPT_HASHSIZE - 4);
    assert(user.sesskey_salt != NULL);
    assert(strlen(user.sesskey_salt) == (BCRYPT_HASHSIZE - 4)/2 - 1);

    assert(0 == bcrypt_hashpw(userProvidedSessionKey, user.sesskey_salt, userProvidedSessionKeyHash));
    assert(userProvidedSessionKeyHash[BCRYPT_HASHSIZE - 4] == '\0');
    assert(strlen(userProvidedSessionKeyHash) == BCRYPT_HASHSIZE - 4);
    LOG_V("__lsapi_endpoint_inventory_put: user-provided session key: %s", userProvidedSessionKey);
    LOG_V("__lsapi_endpoint_inventory_put: userProvidedSessionKeyHash: %s, user.sesskey_hash: %s, user.sesskey_salt: %s", userProvidedSessionKeyHash, user.sesskey_hash, user.sesskey_salt);

    if (0 != strcmp(userProvidedSessionKeyHash, user.sesskey_hash)) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Invalid session key");
    }

    // create inventory item + get inventory item data from database so that we can use it for the http response
    db_inventory_item_t inventoryItem;
    char* rgid_str = __lsapi_itoa(rgid);
    char* lid_str = __lsapi_itoa(lid);
    if (0 != db_inventory_insert_ret(pLsapi->pDb, rgid_str, dadd, dexp, lid_str, epc, apwd, kpwd, &inventoryItem)) {
        yyjson_doc_free(pJson);
        free(rgid_str);
        free(lid_str);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to create inventory item");
    }
    free(rgid_str);
    free(lid_str);

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Inventory item created successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add inventory item data as sub-object
    yyjson_mut_val* pInventoryItem = yyjson_mut_obj(pJsonResp);
    yyjson_mut_obj_add_int(pJsonResp, pInventoryItem, "inventory_item_id", inventoryItem.inventory_id);
    yyjson_mut_obj_add_int(pJsonResp, pInventoryItem, "reagent_id", inventoryItem.reagent_id);
    yyjson_mut_obj_add_str(pJsonResp, pInventoryItem, "date_added", inventoryItem.date_added);
    yyjson_mut_obj_add_str(pJsonResp, pInventoryItem, "date_expire", inventoryItem.date_expire);
    yyjson_mut_obj_add_int(pJsonResp, pInventoryItem, "lab_id", inventoryItem.lab_id);
    yyjson_mut_obj_add_str(pJsonResp, pInventoryItem, "epc", inventoryItem.epc);
    // add inventory item object to root
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "inventory_item", pInventoryItem);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_doc_free(pJson);
    yyjson_mut_doc_free(pJsonResp);
    db_user_free(&user);
    return 0;
}

static int __lsapi_endpoint_inventory_post_action_embody(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi, const int iid) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    assert(pLsapi != NULL);
    assert(iid >= 0);
    // Obtain inventory item data
    db_inventory_item_t inventoryItem;
    char* iid_str = __lsapi_itoa(iid);
    int rv = db_inventory_get_by_id(pLsapi->pDb, iid_str, &inventoryItem);
    free(iid_str);
    if (0 != rv) {
        if (rv == -2) {
            return __lsapi_endpoint_error(pReq, 404, "Not Found", "Inventory item not found");
        } else {
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get inventory item data from database");
        }
    }
    // Obtain lab data
    // TODO use a single query to optimize database load?
    db_lab_t lab;
    char* lid_str = __lsapi_itoa(inventoryItem.lab_id);
    rv = db_lab_get_by_id(pLsapi->pDb, lid_str, &lab);
    free(lid_str);
    if (0 != rv) {
        if (rv == -2) {
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Lab not found (unexpected issue !)");
        } else {
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get lab data from database");
        }
    }
    
    //Communicate with on-premise server
    oph_t* pOph = oph_create(lab.host, lab.lab_key);
    assert(pOph != NULL);
    rv = oph_trigger_embodiment(pOph, inventoryItem.epc, inventoryItem.apwd, inventoryItem.kpwd);
    switch (rv) {
        case 0:
            break;
        case -1:
            oph_destroy(pOph);
            db_lab_free(&lab);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to create request to on-premise server, because the server ran out of memory (poor server)");
        case -2:
            oph_destroy(pOph);
            db_lab_free(&lab);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to create request to on-premise server, because of a network issue");
        case -3:
            oph_destroy(pOph);
            db_lab_free(&lab);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to create request to on-premise server, because of a missing host response");
        case -4:
            oph_destroy(pOph);
            db_lab_free(&lab);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to create request to on-premise server, because of an invalid host response (-4)");
        case -5:
            oph_destroy(pOph);
            db_lab_free(&lab);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to create request to on-premise server, because of an invalid host response (-5)");
        case -6:
            oph_destroy(pOph);
            db_lab_free(&lab);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to create request to on-premise server, because of an invalid host response (-6)");
        case -7:
            oph_destroy(pOph);
            db_lab_free(&lab);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to create request to on-premise server, because of an invalid host response (-7)");
        case -8:
            oph_destroy(pOph);
            db_lab_free(&lab);
            return __lsapi_endpoint_error(pReq, 503, "Service Unavailable", "Resource temporarily unavailable");
        default:
            oph_destroy(pOph);
            db_lab_free(&lab);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to create request to on-premise server, because of an unknown issue (unexpected)");
    }
    oph_destroy(pOph);

    // update inventory item data in database
    iid_str = __lsapi_itoa(iid);
    rv = db_inventory_set_embodied(pLsapi->pDb, iid_str);
    free(iid_str);
    if (0 != rv) {
        LOG_E("__lsapi_endpoint_inventory_post_action_embody: Failed to update inventory item data in database. Data may be inconsistent with on-premise server (embodiment was successful...)!!! Manual intervention required."); // TODO Handle this case gracefully if possible? / Send an email to admin?
        db_lab_free(&lab);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to update inventory item data in database. Data may be inconsistent with on-premise server (embodiment was successful...). Please contact the administrator.");
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Inventory item embodied successfully";

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
    yyjson_mut_doc_free(pJsonResp);
    db_lab_free(&lab);
    return 0;
}

static int __lsapi_endpoint_inventory_post_action_print(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi, const int iid) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    assert(pLsapi != NULL);
    assert(iid >= 0);
    return __lsapi_endpoint_error(pReq, 501, "Not Implemented", "Not Implemented");
}

// curl -X POST -d '{"action": "<embody|print>", "iid": <reagid>, "username":"<username>", "session_key":"<sesskey>"}'
static int __lsapi_endpoint_inventory_post(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
    yyjson_val* pAction = yyjson_obj_get(pRoot, "action");
    if (pAction == NULL || !yyjson_is_str(pAction)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid action");
    }
    yyjson_val* pIid = yyjson_obj_get(pRoot, "iid");
    if (pIid == NULL || !yyjson_is_int(pIid)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid iid (inventory item id)");
    }
    yyjson_val* pUsername = yyjson_obj_get(pRoot, "username");
    if (pUsername == NULL || !yyjson_is_str(pUsername)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid username");
    }
    yyjson_val* pSessionKey = yyjson_obj_get(pRoot, "session_key");
    if (pSessionKey == NULL || !yyjson_is_str(pSessionKey)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid session_key in request body");
    }

    const char* action = yyjson_get_str(pAction);
    const int iid = yyjson_get_int(pIid);
    const char* username = yyjson_get_str(pUsername);
    const char* userProvidedSessionKey = yyjson_get_str(pSessionKey);

    assert(action != NULL && username != NULL && iid >= 0 && userProvidedSessionKey != NULL);

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

    if (0 == strlen(user.sesskey_hash)) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Session key not set");
    }

    // verify session key
    char userProvidedSessionKeyHash[BCRYPT_HASHSIZE];
    assert(user.sesskey_hash != NULL);
    assert(strlen(user.sesskey_hash) == BCRYPT_HASHSIZE - 4);
    assert(user.sesskey_salt != NULL);
    assert(strlen(user.sesskey_salt) == (BCRYPT_HASHSIZE - 4)/2 - 1);

    assert(0 == bcrypt_hashpw(userProvidedSessionKey, user.sesskey_salt, userProvidedSessionKeyHash));
    assert(userProvidedSessionKeyHash[BCRYPT_HASHSIZE - 4] == '\0');
    assert(strlen(userProvidedSessionKeyHash) == BCRYPT_HASHSIZE - 4);
    LOG_V("__lsapi_endpoint_inventory_post: user-provided session key: %s", userProvidedSessionKey);
    LOG_V("__lsapi_endpoint_inventory_post: userProvidedSessionKeyHash: %s, user.sesskey_hash: %s, user.sesskey_salt: %s", userProvidedSessionKeyHash, user.sesskey_hash, user.sesskey_salt);

    if (0 != strcmp(userProvidedSessionKeyHash, user.sesskey_hash)) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Invalid session key");
    }

    //action strategy
    if (0 == strcmp(action, "embody")) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_inventory_post_action_embody(pH2oHandler, pReq, pLsapi, iid);
    } else if (0 == strcmp(action, "print")) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_inventory_post_action_print(pH2oHandler, pReq, pLsapi, iid);
    } else {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid action");
    }
}

static int __lsapi_endpoint_inventory_get(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    return __lsapi_endpoint_x_get(pH2oHandler, pReq, pLsapi, __LSAPI_X_TYPE_INVENTORY_ITEM);
}

int lsapi_endpoint_inventory(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);
    if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("PUT"))) {
        return __lsapi_endpoint_inventory_put(pH2oHandler, pReq, pLsapi);
    } else if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("POST"))) {
        return __lsapi_endpoint_inventory_post(pH2oHandler, pReq, pLsapi);
    } else if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("GET"))) {
        return __lsapi_endpoint_inventory_get(pH2oHandler, pReq, pLsapi);
    } else {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
}

// curl -X POST -d {"filter": "<none|rid|lid|epc|fid|emb>", "value": "<value>", "p_offset": <p_offset>, "p_size": <p_size>}' http://localhost:8080/api/inventory-items
static int __lsapi_endpoint_inventory_items_post(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
    yyjson_val* pFilter = yyjson_obj_get(pRoot, "filter");
    if (pFilter == NULL || !yyjson_is_str(pFilter)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid filter");
    }
    yyjson_val* pValue = yyjson_obj_get(pRoot, "value");
    if (pValue == NULL || !yyjson_is_str(pValue)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid value");
    }
    yyjson_val* pPOffset = yyjson_obj_get(pRoot, "p_offset");
    if (pPOffset == NULL || !yyjson_is_int(pPOffset)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid p_offset");
    }
    yyjson_val* pPSize = yyjson_obj_get(pRoot, "p_size");
    if (pPSize == NULL || !yyjson_is_int(pPSize)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid p_size");
    }

    const char* filter = yyjson_get_str(pFilter);
    const char* value = yyjson_get_str(pValue);
    const int page_offset = yyjson_get_int(pPOffset);
    const int page_size = yyjson_get_int(pPSize);

    assert(filter != NULL && value != NULL);
    if (page_offset < 0 || page_size < 0) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid p_offset or p_size");
    }

    // get inventory items data from database so that we can use it for the http response
    db_inventory_item_t* inventoryItems = NULL;
    int inventoryItemsCount = 0;

    char* page_offset_str = __lsapi_itoa(page_offset);
    char* page_size_str = __lsapi_itoa(page_size);
    db_inventory_item_filter_type_t filter_type = DB_INVENTORY_ITEM_FILTER_TYPE_NONE;
    if (0 == strcmp(filter, "none")) {
        filter_type = DB_INVENTORY_ITEM_FILTER_TYPE_NONE;
    } else if (0 == strcmp(filter, "rid")) {
        filter_type = DB_INVENTORY_ITEM_FILTER_TYPE_REAGENT_ID;
    } else if (0 == strcmp(filter, "lid")) {
        filter_type = DB_INVENTORY_ITEM_FILTER_TYPE_LAB_ID;
    } else if (0 == strcmp(filter, "epc")) {
        filter_type = DB_INVENTORY_ITEM_FILTER_TYPE_EPC;
    } else if (0 == strcmp(filter, "fid")) {
        filter_type = DB_INVENTORY_ITEM_FILTER_TYPE_FACULTY_ID;
    } else if (0 == strcmp(filter, "emb")) {
        filter_type = DB_INVENTORY_ITEM_FILTER_TYPE_IS_EMBODIED;
    } else {
        yyjson_doc_free(pJson);
        free(page_offset_str);
        free(page_size_str);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid filter");
    }
    int rv = db_inventory_items_read_page_filtered(pLsapi->pDb, page_offset_str, page_size_str, &inventoryItems, &inventoryItemsCount, filter_type, value);
    free(page_offset_str);
    free(page_size_str);
    if (0 != rv) {
        if (rv == -2) {
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 404, "Not Found", "No inventory items found");
        } else {
            LOG_E("__lsapi_endpoint_inventory_items_post: Failed to get inventory items data from database (db_inventory_items_read_page_filtered returned %d)", rv);
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get inventory items data from database");
        }
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Inventory items data retrieved successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add inventory items data as array
    yyjson_mut_val* pInventoryItems = yyjson_mut_arr(pJsonResp);
    for (int i = 0; i < inventoryItemsCount; i++) {
        yyjson_mut_val* pInventoryItem = yyjson_mut_obj(pJsonResp);
        yyjson_mut_obj_add_int(pJsonResp, pInventoryItem, "inventory_item_id", inventoryItems[i].inventory_id);
        yyjson_mut_obj_add_int(pJsonResp, pInventoryItem, "reagent_id", inventoryItems[i].reagent_id);
        yyjson_mut_obj_add_str(pJsonResp, pInventoryItem, "date_added", inventoryItems[i].date_added);
        yyjson_mut_obj_add_str(pJsonResp, pInventoryItem, "date_expire", inventoryItems[i].date_expire);
        yyjson_mut_obj_add_int(pJsonResp, pInventoryItem, "lab_id", inventoryItems[i].lab_id);
        yyjson_mut_obj_add_str(pJsonResp, pInventoryItem, "epc", inventoryItems[i].epc);
        yyjson_mut_obj_add_int(pJsonResp, pInventoryItem, "is_embodied", inventoryItems[i].is_embodied);
        yyjson_mut_obj_add_int(pJsonResp, pInventoryItem, "basepoint_id", inventoryItems[i].basepoint_id);
        yyjson_mut_arr_add_val(pInventoryItems, pInventoryItem);
    }
    // add inventory items array to root
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "inventory-items", pInventoryItems);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_doc_free(pJson);
    yyjson_mut_doc_free(pJsonResp);
    assert(inventoryItems != NULL);
    for (int i = 0; i < inventoryItemsCount; i++) {
        db_inventory_item_free(&inventoryItems[i]);
    }
    free(inventoryItems);
    return 0;
}

// gets total number of inventory items in the database
static int __lsapi_endpoint_inventory_items_get(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    assert(pLsapi != NULL);
    int inventoryItemsCount = 0;
    int rv = db_inventory_items_get_total_count(pLsapi->pDb, &inventoryItemsCount);
    if (0 != rv) {
        LOG_E("__Lsapi_endpoint_inventory_items_get: Failed to get total count of inventory items from database (db_inventory_items_get_total_count returned %d)", rv);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get total count of inventory items from database");
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Total number of inventory items retrieved successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    yyjson_mut_obj_add_int(pJsonResp, pRootResp, "inventory_items_count", inventoryItemsCount);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);
    
    free((void*)respText);
    yyjson_mut_doc_free(pJsonResp);
    return 0;
}

int lsapi_endpoint_inventory_items(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);
    
    if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("POST"))) {
        return __lsapi_endpoint_inventory_items_post(pH2oHandler, pReq, pLsapi);
    } else if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("GET"))) {
        return __lsapi_endpoint_inventory_items_get(pH2oHandler, pReq, pLsapi);
    } else {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
}

/**
 * @brief Obtain IP address from provided URL.
 * @note Uses libcurl under the hood.
 * @note Caller is responsible for freeing *pIp_out after use.
 */
static int __lsapi_resolve_url(const char* url_in, char** pIp_out) {
    assert(url_in != NULL);
    assert(pIp_out != NULL);
    CURL* pCurl = curl_easy_init();
    assert(pCurl != NULL);
    assert(CURLE_OK == curl_easy_setopt(pCurl, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4));
    assert(CURLE_OK == curl_easy_setopt(pCurl, CURLOPT_URL, url_in));
    assert(CURLE_OK == curl_easy_setopt(pCurl, CURLOPT_NOBODY, 1L)); // Do not fetch the body
    CURLcode res = curl_easy_perform(pCurl);
    if (res != CURLE_OK) {
        curl_easy_cleanup(pCurl);
        return -1;
    }
    // Get the IP address
    char* ip = NULL;
    assert(CURLE_OK == curl_easy_getinfo(pCurl, CURLINFO_PRIMARY_IP, &ip));
    *pIp_out = p_strdup(ip);
    
    curl_easy_cleanup(pCurl);
    return 0;
}

// curl -X POST -d '{"host": "<host>", "lbtoken": "<lbtoken>"}'
static int __lsapi_endpoint_inven_ld_post(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
    yyjson_val* pHost = yyjson_obj_get(pRoot, "host");
    if (pHost == NULL || !yyjson_is_str(pHost)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid host param");
    }
    yyjson_val* pLbToken = yyjson_obj_get(pRoot, "lbtoken");
    if (pLbToken == NULL || !yyjson_is_str(pLbToken)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid lbtoken (lab token)");
    }

    const char* host = yyjson_get_str(pHost);
    const char* lbToken = yyjson_get_str(pLbToken);
    assert(host != NULL);
    assert(lbToken != NULL);

    // Obtain client IP address
    struct sockaddr sa;
    pReq->conn->callbacks->get_peername(pReq->conn, &sa);
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

    // Check if host param matches client IP address
    char* resolved_ip = NULL;
    if (0 != __lsapi_resolve_url(host, &resolved_ip)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to resolve URL provided in host param");
    }
    assert(resolved_ip != NULL);
    if (0 != strcmp(ip, resolved_ip)) {
        free(resolved_ip);
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Host param does not match client IP address");
    }
    free(resolved_ip);

    // get lab data from database so that we can use it for lbtoken verification and querying inventory
    db_lab_t lab;
    int rv = db_lab_get_by_host(pLsapi->pDb, host, &lab);
    if (0 != rv) {
        if (rv == -2) {
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 404, "Not Found", "Lab not found");
        } else {
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get lab data from database");
        }
    }

    //Verify lbtoken
    char lbTokenHash[BCRYPT_HASHSIZE];
    assert(lab.bearer_token_hash != NULL);
    assert(strlen(lab.bearer_token_hash) == BCRYPT_HASHSIZE - 4);
    assert(lab.bearer_token_salt != NULL);
    assert(strlen(lab.bearer_token_salt) == (BCRYPT_HASHSIZE - 4)/2 - 1);
    assert(0 == bcrypt_hashpw(lbToken, lab.bearer_token_salt, lbTokenHash));
    assert(lbTokenHash[BCRYPT_HASHSIZE - 4] == '\0');
    assert(strlen(lbTokenHash) == BCRYPT_HASHSIZE - 4);
    LOG_V("__lsapi_endpoint_inven_ld_post: lab-provided bearer token: %s", lbToken);
    LOG_V("__lsapi_endpoint_invm_put: lbTokenHash (lab-provided): %s, lab.bearer_token_hash: %s, lab.bearer_token_salt: %s", lbTokenHash, lab.bearer_token_hash, lab.bearer_token_salt);

    if (0 != strcmp(lbTokenHash, lab.bearer_token_hash)) {
        yyjson_doc_free(pJson);
        db_lab_free(&lab);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Invalid lbtoken");
    }

    // get inventory data from database
    db_inventory_item_t* pInventoryItems = NULL;
    size_t nInventoryItems = 0;
    char* lid_str = __lsapi_itoa(lab.lab_id);
    rv = db_inventory_get_by_lab_id_filter_embodied(pLsapi->pDb, lid_str, 1, &pInventoryItems, &nInventoryItems);
    if ((0 != rv) && (-2 != rv)) {
        yyjson_doc_free(pJson);
        db_lab_free(&lab);
        free(lid_str);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get inventory data from database");
    }
    free(lid_str);
    if (-2 == rv) {
        LOG_D("__lsapi_endpoint_inven_ld_post: No inventory items found for lab %d", lab.lab_id);
        assert(nInventoryItems == 0);
        assert(pInventoryItems == NULL);
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Inventory load successful";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", "success");
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", "Inventory load successful");
    yyjson_mut_val* pInventoryItemsArr = yyjson_mut_arr(pJsonResp);
    for (size_t i = 0; i < nInventoryItems; i++) {
        yyjson_mut_val* pInventoryItem = yyjson_mut_obj(pJsonResp);
        yyjson_mut_obj_add_str(pJsonResp, pInventoryItem, "epc", pInventoryItems[i].epc);
        yyjson_mut_obj_add_str(pJsonResp, pInventoryItem, "apwd", pInventoryItems[i].apwd);
        yyjson_mut_obj_add_str(pJsonResp, pInventoryItem, "kpwd", pInventoryItems[i].kpwd);
        yyjson_mut_arr_add_val(pInventoryItemsArr, pInventoryItem);
    }
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "inventory", pInventoryItemsArr);
    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_doc_free(pJson);
    yyjson_mut_doc_free(pJsonResp);
    db_lab_free(&lab);
    if (pInventoryItems != NULL) {
        for (size_t i = 0; i < nInventoryItems; i++) {
            if (pInventoryItems[i].epc == NULL || pInventoryItems[i].apwd == NULL || pInventoryItems[i].kpwd == NULL) {
                assert(0); //just a sanity check
            }
            db_inventory_item_free(&pInventoryItems[i]);
        }
        free(pInventoryItems);
    }
    return 0;
}

int lsapi_endpoint_inven_ld(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);
    if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("POST"))) {
        return __lsapi_endpoint_inven_ld_post(pH2oHandler, pReq, pLsapi);
    } else {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
}

// curl -X PUT -d '{"aname": "<antenna name>", "info": "<antenna info>", "k": <k>, "lid": <lab id>, "username": "<username>", "session_key": "<sesskey>"}'
static int __lsapi_endpoint_antenna_put(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
    yyjson_val* pAName = yyjson_obj_get(pRoot, "aname");
    if (pAName == NULL || !yyjson_is_str(pAName)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid aname (antenna name)");
    }
    yyjson_val* pInfo = yyjson_obj_get(pRoot, "info");
    if (pInfo == NULL || !yyjson_is_str(pInfo)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid info (antenna info)");
    }
    yyjson_val* pK = yyjson_obj_get(pRoot, "k");
    if (pK == NULL || !yyjson_is_int(pK)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid k (antenna k)");
    }
    yyjson_val* pLid = yyjson_obj_get(pRoot, "lid");
    if (pLid == NULL || !yyjson_is_int(pLid)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid lid (lab id)");
    }
    yyjson_val* pUsername = yyjson_obj_get(pRoot, "username");
    if (pUsername == NULL || !yyjson_is_str(pUsername)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid username");
    }
    yyjson_val* pSessionKey = yyjson_obj_get(pRoot, "session_key");
    if (pSessionKey == NULL || !yyjson_is_str(pSessionKey)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid session_key in request body");
    }

    const char* aName = yyjson_get_str(pAName);
    const char* info = yyjson_get_str(pInfo);
    int k = yyjson_get_int(pK);
    int lid = yyjson_get_int(pLid);
    const char* username = yyjson_get_str(pUsername);
    const char* userProvidedSessionKey = yyjson_get_str(pSessionKey);

    assert(aName != NULL && info != NULL && k >= 0 && lid >= 0 && username != NULL && userProvidedSessionKey != NULL);

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

    if (0 == strlen(user.sesskey_hash)) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Session key not set");
    }

    // verify session key
    char userProvidedSessionKeyHash[BCRYPT_HASHSIZE];
    assert(user.sesskey_hash != NULL);
    assert(strlen(user.sesskey_hash) == BCRYPT_HASHSIZE - 4);
    assert(user.sesskey_salt != NULL);
    assert(strlen(user.sesskey_salt) == (BCRYPT_HASHSIZE - 4)/2 - 1);

    assert(0 == bcrypt_hashpw(userProvidedSessionKey, user.sesskey_salt, userProvidedSessionKeyHash));
    assert(userProvidedSessionKeyHash[BCRYPT_HASHSIZE - 4] == '\0');
    assert(strlen(userProvidedSessionKeyHash) == BCRYPT_HASHSIZE - 4);
    LOG_V("__lsapi_endpoint_antenna_put: user-provided session key: %s", userProvidedSessionKey);
    LOG_V("__lsapi_endpoint_antenna_put: userProvidedSessionKeyHash: %s, user.sesskey_hash: %s, user.sesskey_salt: %s", userProvidedSessionKeyHash, user.sesskey_hash, user.sesskey_salt);

    if (0 != strcmp(userProvidedSessionKeyHash, user.sesskey_hash)) {
        yyjson_doc_free(pJson);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Invalid session key");
    }

    // create antenna + get antenna data from database so that we can use it for the http response
    db_antenna_t antenna;
    char* k_str = __lsapi_itoa(k);
    char* lid_str = __lsapi_itoa(lid);
    if (0 != db_antenna_insert_ret(pLsapi->pDb, aName, info, k_str, lid_str, &antenna)) {
        yyjson_doc_free(pJson);
        free(k_str);
        free(lid_str);
        db_user_free(&user);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to create antenna");
    }
    free(k_str);
    free(lid_str);

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Antenna created successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add antenna data as sub-object
    yyjson_mut_val* pAntenna = yyjson_mut_obj(pJsonResp);
    yyjson_mut_obj_add_int(pJsonResp, pAntenna, "antenna_id", antenna.antenna_id);
    yyjson_mut_obj_add_str(pJsonResp, pAntenna, "name", antenna.name);
    yyjson_mut_obj_add_str(pJsonResp, pAntenna, "info", antenna.info);
    yyjson_mut_obj_add_int(pJsonResp, pAntenna, "k", antenna.k);
    yyjson_mut_obj_add_int(pJsonResp, pAntenna, "lab_id", antenna.lab_id);
    // add antenna object to root
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "antenna", pAntenna);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_doc_free(pJson);
    yyjson_mut_doc_free(pJsonResp);
    db_user_free(&user);
    db_antenna_free(&antenna);
    return 0;
}

// // Obtain a list of antennas for given lab's bearer token
// // curl -X POST -d '{"btoken": "<bearer token>"}'
// static int __lsapi_endpoint_antenna_post(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
//     assert(pH2oHandler != NULL);
//     assert(pReq != NULL);
//     assert(pLsapi != NULL);
//     return __lsapi_endpoint_error(pReq, 501, "Not Implemented", "Not Implemented");
// }

static int __lsapi_endpoint_antenna_get(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
    return __lsapi_endpoint_x_get(pH2oHandler, pReq, pLsapi, __LSAPI_X_TYPE_ANTENNA);
}

int lsapi_endpoint_antenna(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);
    if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("PUT"))) {
        return __lsapi_endpoint_antenna_put(pH2oHandler, pReq, pLsapi);
    // } else if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("POST"))) {
    //     return __lsapi_endpoint_antenna_post(pH2oHandler, pReq, pLsapi);
    } else if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("GET"))) {
        return __lsapi_endpoint_antenna_get(pH2oHandler, pReq, pLsapi);
    } else {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
}

// curl -X PUT -d '{"t": "<t>", "epc": "<epc>", "an": <antno>, "rxss": <rx signal strength>, "rxrate": <read rate>, "txp": <tx power>, "rxlat": <read latency>, "mtype": <measurement type>, "rkt": <rotator ktheta>, "rkp": <rotator kphi>, "lbtoken": "<lab bearer token>"}'
static int __lsapi_endpoint_invm_put(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
    yyjson_val* pT = yyjson_obj_get(pRoot, "t");
    if (pT == NULL || !yyjson_is_str(pT)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid t (timestamp)");
    }
    yyjson_val* pEpc = yyjson_obj_get(pRoot, "epc");
    if (pEpc == NULL || !yyjson_is_str(pEpc)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid epc");
    }
    yyjson_val* pAn = yyjson_obj_get(pRoot, "an");
    if (pAn == NULL || !yyjson_is_int(pAn)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid an (antno)");
    }
    yyjson_val* pRxss = yyjson_obj_get(pRoot, "rxss");
    if (pRxss == NULL || !yyjson_is_int(pRxss)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rxss (rx signal strength)");
    }
    yyjson_val* pRxrate = yyjson_obj_get(pRoot, "rxrate");
    if (pRxrate == NULL || !yyjson_is_int(pRxrate)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rxrate (read rate)");
    }
    yyjson_val* pTxp = yyjson_obj_get(pRoot, "txp");
    if (pTxp == NULL || !yyjson_is_int(pTxp)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid txp (tx power)");
    }
    yyjson_val* pRxlat = yyjson_obj_get(pRoot, "rxlat");
    if (pRxlat == NULL || !yyjson_is_int(pRxlat)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rxlat (read latency)");
    }
    yyjson_val* pMType = yyjson_obj_get(pRoot, "mtype");
    if (pMType == NULL || !yyjson_is_int(pMType)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid mtype (measurement type)");
    }
    yyjson_val* pRkt = yyjson_obj_get(pRoot, "rkt");
    if (pRkt == NULL || !yyjson_is_int(pRkt)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rkt (rotator ktheta)");
    }
    yyjson_val* pRkp = yyjson_obj_get(pRoot, "rkp");
    if (pRkp == NULL || !yyjson_is_int(pRkp)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rkp (rotator kphi)");
    }
    yyjson_val* pLbToken = yyjson_obj_get(pRoot, "lbtoken");
    if (pLbToken == NULL || !yyjson_is_str(pLbToken)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid lbtoken (lab bearer token)");
    }

    const char* t = yyjson_get_str(pT);
    const char* epc = yyjson_get_str(pEpc);
    int an = yyjson_get_int(pAn);
    int rxss = yyjson_get_int(pRxss);
    int rxrate = yyjson_get_int(pRxrate);
    int txp = yyjson_get_int(pTxp);
    int rxlat = yyjson_get_int(pRxlat);
    int mtype = yyjson_get_int(pMType);
    int rkt = yyjson_get_int(pRkt);
    int rkp = yyjson_get_int(pRkp);
    const char* lbToken = yyjson_get_str(pLbToken);

    //assert(t != NULL && epc != NULL && an >= 0 && rxss >= 0 && rxrate >= 0 && txp >= 0 && rxlat >= 0 && mtype >= 0 && rkt >= 0 && rkp >= 0 && lbToken != NULL);
    assert(t != NULL);
    assert(epc != NULL);
    assert(an >= 0);
    assert(rxss >= 0);
    assert(rxrate >= 0 || rxrate == -1);
    assert(txp >= 0);
    assert(rxlat >= 0 || rxlat == -1);
    assert(mtype >= 0);
    assert(rkt >= 0 || rkt == -1);
    assert(rkp >= 0 || rkp == -1);
    assert(lbToken != NULL);

    // get lab data from database so that we can verify the lab bearer token
    db_lab_t lab;
    int rv = db_lab_get_by_epc(pLsapi->pDb, epc, &lab);
    if (0 != rv) {
        if (rv == -2) {
            LOG_W("__lsapi_endpoint_invm_put: EPC %s does not match any lab", epc);
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 404, "Not Found", "The given epc does not match any lab");
        } else {
            yyjson_doc_free(pJson);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get lab data from database");
        }
    }

    // verify lab bearer token
    char lbTokenHash[BCRYPT_HASHSIZE];
    assert(lab.bearer_token_hash != NULL);
    assert(strlen(lab.bearer_token_hash) == BCRYPT_HASHSIZE - 4);
    assert(lab.bearer_token_salt != NULL);
    assert(strlen(lab.bearer_token_salt) == (BCRYPT_HASHSIZE - 4)/2 - 1);

    assert(0 == bcrypt_hashpw(lbToken, lab.bearer_token_salt, lbTokenHash));
    assert(lbTokenHash[BCRYPT_HASHSIZE - 4] == '\0');
    assert(strlen(lbTokenHash) == BCRYPT_HASHSIZE - 4);
    LOG_V("__lsapi_endpoint_invm_put: lab-provided bearer token: %s", lbToken);
    LOG_V("__lsapi_endpoint_invm_put: lbTokenHash (lab-provided): %s, lab.bearer_token_hash: %s, lab.bearer_token_salt: %s", lbTokenHash, lab.bearer_token_hash, lab.bearer_token_salt);

    if (0 != strcmp(lbTokenHash, lab.bearer_token_hash)) {
        yyjson_doc_free(pJson);
        db_lab_free(&lab);
        return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Invalid lab bearer token");
    }

    // insert inventory measurement + get inventory measurement data from database so that we can use it for the http response
    db_invm_t invm;
    char* an_str = __lsapi_itoa(an);
    char* rxss_str = __lsapi_itoa(rxss);
    char* rxrate_str = __lsapi_itoa(rxrate);
    char* txp_str = __lsapi_itoa(txp);
    char* rxlat_str = __lsapi_itoa(rxlat);
    char* mtype_str = __lsapi_itoa(mtype);
    char* rkt_str = __lsapi_itoa(rkt);
    char* rkp_str = __lsapi_itoa(rkp);
    if (0 != db_invm_insert_ret(pLsapi->pDb, t, epc, an_str, rxss_str, rxrate_str, txp_str, rxlat_str, mtype_str, rkt_str, rkp_str, &invm)) {
        yyjson_doc_free(pJson);
        free(an_str);
        free(rxss_str);
        free(rxrate_str);
        free(txp_str);
        free(rxlat_str);
        free(mtype_str);
        free(rkt_str);
        free(rkp_str);
        db_lab_free(&lab);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to insert inventory measurement");
    }
    free(an_str);
    free(rxss_str);
    free(rxrate_str);
    free(txp_str);
    free(rxlat_str);
    free(mtype_str);
    free(rkt_str);
    free(rkp_str);

    static h2o_generator_t generator = {NULL, NULL}; // TODO should we really have it static?
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Inventory measurement inserted successfully";

    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add inventory measurement data as sub-object
    yyjson_mut_val* pInvm = yyjson_mut_obj(pJsonResp);
    yyjson_mut_obj_add_str(pJsonResp, pInvm, "t", invm.time);
    yyjson_mut_obj_add_str(pJsonResp, pInvm, "epc", invm.inventory_epc);
    yyjson_mut_obj_add_int(pJsonResp, pInvm, "an", invm.antno);
    yyjson_mut_obj_add_int(pJsonResp, pInvm, "rx_signal_strength", invm.rx_signal_strength);
    yyjson_mut_obj_add_int(pJsonResp, pInvm, "read_rate", invm.read_rate);
    yyjson_mut_obj_add_int(pJsonResp, pInvm, "tx_power", invm.tx_power);
    yyjson_mut_obj_add_int(pJsonResp, pInvm, "read_latency", invm.read_latency);
    yyjson_mut_obj_add_int(pJsonResp, pInvm, "measurement_type", invm.measurement_type);
    yyjson_mut_obj_add_int(pJsonResp, pInvm, "rotator_ktheta", invm.rotator_ktheta);
    yyjson_mut_obj_add_int(pJsonResp, pInvm, "rotator_kphi", invm.rotator_kphi);
    // add inventory measurement object to root
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "inventory_measurement", pInvm);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_doc_free(pJson);
    yyjson_mut_doc_free(pJsonResp);
    db_lab_free(&lab);
    db_invm_free(&invm);
    return 0;
}

#define __LSAPI_ENDPOINT_INVM_BULK_PUT_MAX_INVMS 1024

static int __lsapi_endpoint_invm_bulk_put_optimized(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
    yyjson_val* pNInvms = yyjson_obj_get(pRoot, "n_invms");
    if (pNInvms == NULL || !yyjson_is_int(pNInvms)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid n_invms (number of inventory measurements)");
    }
    yyjson_val* pLbToken = yyjson_obj_get(pRoot, "lbtoken");
    if (pLbToken == NULL || !yyjson_is_str(pLbToken)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid lbtoken (lab bearer token)");
    }

    const int nInvms = yyjson_get_int(pNInvms);
    const char* lbToken = yyjson_get_str(pLbToken);

    yyjson_val* pInvms = yyjson_obj_get(pRoot, "invms");
    if (pInvms == NULL || !yyjson_is_arr(pInvms)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid invms (inventory measurements)");
    }

    if (nInvms <= 0 || nInvms > __LSAPI_ENDPOINT_INVM_BULK_PUT_MAX_INVMS) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Unsupported value of n_invms (number of inventory measurements to insert)");
    }

    //db_invm_t* pInvms_structs = (db_invm_t*)malloc(nInvms * sizeof(db_invm_t));
    // Process each inventory measurement
    yyjson_val* pInvm;
    yyjson_arr_iter iter = yyjson_arr_iter_with(pInvms);

    char** pTimes = (char**)malloc(nInvms * sizeof(char*));
    char** pEpcs = (char**)malloc(nInvms * sizeof(char*));
    char** pAntnos = (char**)malloc(nInvms * sizeof(char*));
    char** pRxsss = (char**)malloc(nInvms * sizeof(char*));
    char** pRxrates = (char**)malloc(nInvms * sizeof(char*));
    char** pTxps = (char**)malloc(nInvms * sizeof(char*));
    char** pRxlats = (char**)malloc(nInvms * sizeof(char*));
    char** pMtypes = (char**)malloc(nInvms * sizeof(char*));
    char** pRkts = (char**)malloc(nInvms * sizeof(char*));
    char** pRkps = (char**)malloc(nInvms * sizeof(char*));

    while ((pInvm = yyjson_arr_iter_next(&iter))) {
        assert(iter.idx-1 >= 0);
        LOG_V("__lsapi_endpoint_invm_bulk_put_optimized: Processing invm, iter.idx=%d", iter.idx);
        if (!(iter.idx-1 < nInvms)) {
            yyjson_doc_free(pJson);
            // for (int i = 0; i < iter.idx-1; i++) {
            //     db_invm_free(&pInvms_structs[i]);
            // }
            // free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Provided more inventory measurements than specified by n_invms");
        }
        if (!yyjson_is_obj(pInvm)) {
            yyjson_doc_free(pJson);
            // for (int i = 0; i < iter.idx-1; i++) {
            //     db_invm_free(&pInvms_structs[i]);
            // }
            // free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid inventory measurement object found in invms array");
        }
        yyjson_val* pT = yyjson_obj_get(pInvm, "t");
        if (pT == NULL || !yyjson_is_str(pT)) {
            yyjson_doc_free(pJson);
            // for (int i = 0; i < iter.idx-1; i++) {
            //     db_invm_free(&pInvms_structs[i]);
            // }
            // free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid t (timestamp)");
        }
        yyjson_val* pEpc = yyjson_obj_get(pInvm, "epc");
        if (pEpc == NULL || !yyjson_is_str(pEpc)) {
            yyjson_doc_free(pJson);
            // for (int i = 0; i < iter.idx-1; i++) {
            //     db_invm_free(&pInvms_structs[i]);
            // }
            // free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid epc");
        }
        yyjson_val* pAn = yyjson_obj_get(pInvm, "an");
        if (pAn == NULL || !yyjson_is_int(pAn)) {
            yyjson_doc_free(pJson);
            // for (int i = 0; i < iter.idx-1; i++) {
            //     db_invm_free(&pInvms_structs[i]);
            // }
            // free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid an (antno)");
        }
        yyjson_val* pRxss = yyjson_obj_get(pInvm, "rxss");
        if (pRxss == NULL || !yyjson_is_int(pRxss)) {
            yyjson_doc_free(pJson);
            // for (int i = 0; i < iter.idx-1; i++) {
            //     db_invm_free(&pInvms_structs[i]);
            // }
            // free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rxss (rx signal strength)");
        }
        yyjson_val* pRxrate = yyjson_obj_get(pInvm, "rxrate");
        if (pRxrate == NULL || !yyjson_is_int(pRxrate)) {
            yyjson_doc_free(pJson);
            // for (int i = 0; i < iter.idx-1; i++) {
            //     db_invm_free(&pInvms_structs[i]);
            // }
            // free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rxrate (read rate)");
        }
        yyjson_val* pTxp = yyjson_obj_get(pInvm, "txp");
        if (pTxp == NULL || !yyjson_is_int(pTxp)) {
            yyjson_doc_free(pJson);
            // for (int i = 0; i < iter.idx-1; i++) {
            //     db_invm_free(&pInvms_structs[i]);
            // }
            // free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid txp (tx power)");
        }
        yyjson_val* pRxlat = yyjson_obj_get(pInvm, "rxlat");
        if (pRxlat == NULL || !yyjson_is_int(pRxlat)) {
            yyjson_doc_free(pJson);
            // for (int i = 0; i < iter.idx-1; i++) {
            //     db_invm_free(&pInvms_structs[i]);
            // }
            // free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rxlat (read latency)");
        }
        yyjson_val* pMType = yyjson_obj_get(pInvm, "mtype");
        if (pMType == NULL || !yyjson_is_int(pMType)) {
            yyjson_doc_free(pJson);
            // for (int i = 0; i < iter.idx-1; i++) {
            //     db_invm_free(&pInvms_structs[i]);
            // }
            // free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid mtype (measurement type)");
        }
        yyjson_val* pRkt = yyjson_obj_get(pInvm, "rkt");
        if (pRkt == NULL || !yyjson_is_int(pRkt)) {
            yyjson_doc_free(pJson);
            // for (int i = 0; i < iter.idx-1; i++) {
            //     db_invm_free(&pInvms_structs[i]);
            // }
            // free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rkt (rotator ktheta)");
        }
        yyjson_val* pRkp = yyjson_obj_get(pInvm, "rkp");
        if (pRkp == NULL || !yyjson_is_int(pRkp)) {
            yyjson_doc_free(pJson);
            // for (int i = 0; i < iter.idx-1; i++) {
            //     db_invm_free(&pInvms_structs[i]);
            // }
            // free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rkp (rotator kphi)");
        }

        const char* t = yyjson_get_str(pT);
        const char* epc = yyjson_get_str(pEpc);
        int an = yyjson_get_int(pAn);
        int rxss = yyjson_get_int(pRxss);
        int rxrate = yyjson_get_int(pRxrate);
        int txp = yyjson_get_int(pTxp);
        int rxlat = yyjson_get_int(pRxlat);
        int mtype = yyjson_get_int(pMType);
        int rkt = yyjson_get_int(pRkt);
        int rkp = yyjson_get_int(pRkp);

        assert(t != NULL);
        assert(epc != NULL);
        assert(an >= 0);
        assert(rxss >= 0);
        assert(rxrate >= 0 || rxrate == -1);
        assert(txp >= 0);
        assert(rxlat >= 0 || rxlat == -1);
        assert(mtype >= 0);
        assert(rkt >= 0 || rkt == -1);
        assert(rkp >= 0 || rkp == -1);

        // SKIP LAB BEARER TOKEN VERIFICATION FOR EACH INVENTORY MEASUREMENT
        // TODO verify lab bearer token only once for the entire bulk request - but how do we do that without performing join for each epc? - JWT, again? Or some JWT-like mechanism involving epc and lab secret?

        char* an_str = __lsapi_itoa(an);
        char* rxss_str = __lsapi_itoa(rxss);
        char* rxrate_str = __lsapi_itoa(rxrate);
        char* txp_str = __lsapi_itoa(txp);
        char* rxlat_str = __lsapi_itoa(rxlat);
        char* mtype_str = __lsapi_itoa(mtype);
        char* rkt_str = __lsapi_itoa(rkt);
        char* rkp_str = __lsapi_itoa(rkp);

        pTimes[iter.idx-1] = p_strdup(t);
        pEpcs[iter.idx-1] = p_strdup(epc);
        pAntnos[iter.idx-1] = an_str;
        pRxsss[iter.idx-1] = rxss_str;
        pRxrates[iter.idx-1] = rxrate_str;
        pTxps[iter.idx-1] = txp_str;
        pRxlats[iter.idx-1] = rxlat_str;
        pMtypes[iter.idx-1] = mtype_str;
        pRkts[iter.idx-1] = rkt_str;
        pRkps[iter.idx-1] = rkp_str;
        
    }

    LOG_V("__lsapi_endpoint_invm_bulk_put_optimized: iter.idx=%d, nInvms=%d", iter.idx, nInvms);
    if (iter.idx != nInvms) {
        yyjson_doc_free(pJson);
        // for (int i = 0; i < iter.idx; i++) {
        //     db_invm_free(&pInvms_structs[i]);
        // }
        // free(pInvms_structs);
        for (int i = 0; i < iter.idx; i++) {
            free(pTimes[i]);
            free(pEpcs[i]);
            free(pAntnos[i]);
            free(pRxsss[i]);
            free(pRxrates[i]);
            free(pTxps[i]);
            free(pRxlats[i]);
            free(pMtypes[i]);
            free(pRkts[i]);
            free(pRkps[i]);
        }
        free(pTimes);
        free(pEpcs);
        free(pAntnos);
        free(pRxsss);
        free(pRxrates);
        free(pTxps);
        free(pRxlats);
        free(pMtypes);
        free(pRkts);
        free(pRkps);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Provided fewer inventory measurements than specified by n_invms");
    }

    // insert inventory measurements
    int rv = db_invm_insert_bulk(pLsapi->pDb, nInvms, (const char**)pTimes, (const char**)pEpcs, (const char**)pAntnos, (const char**)pRxsss, (const char**)pRxrates, (const char**)pTxps, (const char**)pRxlats, (const char**)pMtypes, (const char**)pRkts, (const char**)pRkps);
    if (0 != rv) {
        yyjson_doc_free(pJson);
        // for (int i = 0; i < nInvms; i++) {
        //     db_invm_free(&pInvms_structs[i]);
        // }
        // free(pInvms_structs);
        for (int i = 0; i < nInvms; i++) {
            free(pTimes[i]);
            free(pEpcs[i]);
            free(pAntnos[i]);
            free(pRxsss[i]);
            free(pRxrates[i]);
            free(pTxps[i]);
            free(pRxlats[i]);
            free(pMtypes[i]);
            free(pRkts[i]);
            free(pRkps[i]);
        }
        free(pTimes);
        free(pEpcs);
        free(pAntnos);
        free(pRxsss);
        free(pRxrates);
        free(pTxps);
        free(pRxlats);
        free(pMtypes);
        free(pRkts);
        free(pRkps);
        return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to insert inventory measurements");
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Inventory measurements inserted successfully (bulk, optimized)";

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
    // for (int i = 0; i < nInvms; i++) {
    //     db_invm_free(&pInvms_structs[i]);
    // }
    // free(pInvms_structs);
    for (int i = 0; i < nInvms; i++) {
        free(pTimes[i]);
        free(pEpcs[i]);
        free(pAntnos[i]);
        free(pRxsss[i]);
        free(pRxrates[i]);
        free(pTxps[i]);
        free(pRxlats[i]);
        free(pMtypes[i]);
        free(pRkts[i]);
        free(pRkps[i]);
    }
    free(pTimes);
    free(pEpcs);
    free(pAntnos);
    free(pRxsss);
    free(pRxrates);
    free(pTxps);
    free(pRxlats);
    free(pMtypes);
    free(pRkts);
    free(pRkps);
    return 0;
}

int lsapi_endpoint_invm(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);
    if (!h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("PUT"))) {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
    return __lsapi_endpoint_invm_put(pH2oHandler, pReq, pLsapi);
}

// curl -X PUT -d '{"lbtoken": "<lab bearer token>", "n_invms": <n_invms>, "invms": [
//     {"t": "<t1>", "epc": "<epc1>", "an": <antno1>, "rxss": <rx_signal_strength_1>, "rxrate": <read_rate_1>, "txp": <tx_power_1>, "rxlat": <rxlat1>, "mtype": <mtype1>, "rkt": <rkt1>, "rkp": <rkp1>},
//     {"t": "<t2>", "epc": "<epc2>", "an": <antno2>, "rxss": <rx_signal_strength_2>, "rxrate": <read_rate_2>, "txp": <tx_power_2>, "rxlat": <rxlat2>, "mtype": <mtype2>, "rkt": <rkt2>, "rkp": <rkp2>},
//     ...
//    {"t": "<tn>", "epc": "<epcn>", "an": <antnon>, "rxss": <rx_signal_strength_n>, "rxrate": <read_rate_n>, "txp": <tx_power_n>, "rxlat": <rxlatn>, "mtype": <mtypen>, "rkt": <rktn>, "rkp": <rkpn>}
// ]}'
static int __lsapi_endpoint_invm_bulk_put(h2o_handler_t* pH2oHandler, h2o_req_t* pReq, lsapi_t* pLsapi) {
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
    yyjson_val* pNInvms = yyjson_obj_get(pRoot, "n_invms");
    if (pNInvms == NULL || !yyjson_is_int(pNInvms)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid n_invms (number of inventory measurements)");
    }
    yyjson_val* pLbToken = yyjson_obj_get(pRoot, "lbtoken");
    if (pLbToken == NULL || !yyjson_is_str(pLbToken)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid lbtoken (lab bearer token)");
    }

    const int nInvms = yyjson_get_int(pNInvms);
    const char* lbToken = yyjson_get_str(pLbToken);

    yyjson_val* pInvms = yyjson_obj_get(pRoot, "invms");
    if (pInvms == NULL || !yyjson_is_arr(pInvms)) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid invms (inventory measurements)");
    }

    if (nInvms <= 0 || nInvms > __LSAPI_ENDPOINT_INVM_BULK_PUT_MAX_INVMS) {
        yyjson_doc_free(pJson);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Unsupported value of n_invms (number of inventory measurements to insert)");
    }

    db_invm_t* pInvms_structs = (db_invm_t*)malloc(nInvms * sizeof(db_invm_t));
    // Process each inventory measurement
    yyjson_val* pInvm;
    yyjson_arr_iter iter = yyjson_arr_iter_with(pInvms);
    while ((pInvm = yyjson_arr_iter_next(&iter))) {
        assert(iter.idx-1 >= 0);
        LOG_V("__lsapi_endpoint_invm_bulk_put: Processing invm, iter.idx=%d", iter.idx);
        if (!(iter.idx-1 < nInvms)) {
            yyjson_doc_free(pJson);
            for (int i = 0; i < iter.idx-1; i++) {
                db_invm_free(&pInvms_structs[i]);
            }
            free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Provided more inventory measurements than specified by n_invms");
        }
        if (!yyjson_is_obj(pInvm)) {
            yyjson_doc_free(pJson);
            for (int i = 0; i < iter.idx-1; i++) {
                db_invm_free(&pInvms_structs[i]);
            }
            free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid inventory measurement object found in invms array");
        }
        yyjson_val* pT = yyjson_obj_get(pInvm, "t");
        if (pT == NULL || !yyjson_is_str(pT)) {
            yyjson_doc_free(pJson);
            for (int i = 0; i < iter.idx-1; i++) {
                db_invm_free(&pInvms_structs[i]);
            }
            free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid t (timestamp)");
        }
        yyjson_val* pEpc = yyjson_obj_get(pInvm, "epc");
        if (pEpc == NULL || !yyjson_is_str(pEpc)) {
            yyjson_doc_free(pJson);
            for (int i = 0; i < iter.idx-1; i++) {
                db_invm_free(&pInvms_structs[i]);
            }
            free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid epc");
        }
        yyjson_val* pAn = yyjson_obj_get(pInvm, "an");
        if (pAn == NULL || !yyjson_is_int(pAn)) {
            yyjson_doc_free(pJson);
            for (int i = 0; i < iter.idx-1; i++) {
                db_invm_free(&pInvms_structs[i]);
            }
            free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid an (antno)");
        }
        yyjson_val* pRxss = yyjson_obj_get(pInvm, "rxss");
        if (pRxss == NULL || !yyjson_is_int(pRxss)) {
            yyjson_doc_free(pJson);
            for (int i = 0; i < iter.idx-1; i++) {
                db_invm_free(&pInvms_structs[i]);
            }
            free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rxss (rx signal strength)");
        }
        yyjson_val* pRxrate = yyjson_obj_get(pInvm, "rxrate");
        if (pRxrate == NULL || !yyjson_is_int(pRxrate)) {
            yyjson_doc_free(pJson);
            for (int i = 0; i < iter.idx-1; i++) {
                db_invm_free(&pInvms_structs[i]);
            }
            free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rxrate (read rate)");
        }
        yyjson_val* pTxp = yyjson_obj_get(pInvm, "txp");
        if (pTxp == NULL || !yyjson_is_int(pTxp)) {
            yyjson_doc_free(pJson);
            for (int i = 0; i < iter.idx-1; i++) {
                db_invm_free(&pInvms_structs[i]);
            }
            free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid txp (tx power)");
        }
        yyjson_val* pRxlat = yyjson_obj_get(pInvm, "rxlat");
        if (pRxlat == NULL || !yyjson_is_int(pRxlat)) {
            yyjson_doc_free(pJson);
            for (int i = 0; i < iter.idx-1; i++) {
                db_invm_free(&pInvms_structs[i]);
            }
            free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rxlat (read latency)");
        }
        yyjson_val* pMType = yyjson_obj_get(pInvm, "mtype");
        if (pMType == NULL || !yyjson_is_int(pMType)) {
            yyjson_doc_free(pJson);
            for (int i = 0; i < iter.idx-1; i++) {
                db_invm_free(&pInvms_structs[i]);
            }
            free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid mtype (measurement type)");
        }
        yyjson_val* pRkt = yyjson_obj_get(pInvm, "rkt");
        if (pRkt == NULL || !yyjson_is_int(pRkt)) {
            yyjson_doc_free(pJson);
            for (int i = 0; i < iter.idx-1; i++) {
                db_invm_free(&pInvms_structs[i]);
            }
            free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rkt (rotator ktheta)");
        }
        yyjson_val* pRkp = yyjson_obj_get(pInvm, "rkp");
        if (pRkp == NULL || !yyjson_is_int(pRkp)) {
            yyjson_doc_free(pJson);
            for (int i = 0; i < iter.idx-1; i++) {
                db_invm_free(&pInvms_structs[i]);
            }
            free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid rkp (rotator kphi)");
        }

        const char* t = yyjson_get_str(pT);
        const char* epc = yyjson_get_str(pEpc);
        int an = yyjson_get_int(pAn);
        int rxss = yyjson_get_int(pRxss);
        int rxrate = yyjson_get_int(pRxrate);
        int txp = yyjson_get_int(pTxp);
        int rxlat = yyjson_get_int(pRxlat);
        int mtype = yyjson_get_int(pMType);
        int rkt = yyjson_get_int(pRkt);
        int rkp = yyjson_get_int(pRkp);

        assert(t != NULL);
        assert(epc != NULL);
        assert(an >= 0);
        assert(rxss >= 0);
        assert(rxrate >= 0 || rxrate == -1);
        assert(txp >= 0);
        assert(rxlat >= 0 || rxlat == -1);
        assert(mtype >= 0);
        assert(rkt >= 0 || rkt == -1);
        assert(rkp >= 0 || rkp == -1);

        // get lab data from database so that we can verify the lab bearer token
        db_lab_t lab;
        int rv = db_lab_get_by_epc(pLsapi->pDb, epc, &lab);
        if (0 != rv) {
            if (rv == -2) {
                LOG_W("__lsapi_endpoint_invm_put: EPC %s does not match any lab", epc);
                yyjson_doc_free(pJson);
                for (int i = 0; i < iter.idx-1; i++) {
                    db_invm_free(&pInvms_structs[i]);
                }
                free(pInvms_structs);
                return __lsapi_endpoint_error(pReq, 404, "Not Found", "The given epc does not match any lab");
            } else {
                yyjson_doc_free(pJson);
                for (int i = 0; i < iter.idx-1; i++) {
                    db_invm_free(&pInvms_structs[i]);
                }
                free(pInvms_structs);
                return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to get lab data from database");
            }
        }

        // verify lab bearer token
        char lbTokenHash[BCRYPT_HASHSIZE];
        assert(lab.bearer_token_hash != NULL);
        assert(strlen(lab.bearer_token_hash) == BCRYPT_HASHSIZE - 4);
        assert(lab.bearer_token_salt != NULL);
        assert(strlen(lab.bearer_token_salt) == (BCRYPT_HASHSIZE - 4)/2 - 1);

        assert(0 == bcrypt_hashpw(lbToken, lab.bearer_token_salt, lbTokenHash));
        assert(lbTokenHash[BCRYPT_HASHSIZE - 4] == '\0');
        assert(strlen(lbTokenHash) == BCRYPT_HASHSIZE - 4);
        LOG_V("__lsapi_endpoint_invm_put: lab-provided bearer token: %s", lbToken);
        LOG_V("__lsapi_endpoint_invm_put: lbTokenHash (lab-provided): %s, lab.bearer_token_hash: %s, lab.bearer_token_salt: %s", lbTokenHash, lab.bearer_token_hash, lab.bearer_token_salt);

        if (0 != strcmp(lbTokenHash, lab.bearer_token_hash)) {
            yyjson_doc_free(pJson);
            db_lab_free(&lab);
            for (int i = 0; i < iter.idx-1; i++) {
                db_invm_free(&pInvms_structs[i]);
            }
            free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 403, "Forbidden", "Invalid lab bearer token");
        }

        // insert inventory measurement + get inventory measurement data from database so that we can include it in http response
        char* an_str = __lsapi_itoa(an);
        char* rxss_str = __lsapi_itoa(rxss);
        char* rxrate_str = __lsapi_itoa(rxrate);
        char* txp_str = __lsapi_itoa(txp);
        char* rxlat_str = __lsapi_itoa(rxlat);
        char* mtype_str = __lsapi_itoa(mtype);
        char* rkt_str = __lsapi_itoa(rkt);
        char* rkp_str = __lsapi_itoa(rkp);
        if (0 != db_invm_insert_ret(pLsapi->pDb, t, epc, an_str, rxss_str, rxrate_str, txp_str, rxlat_str, mtype_str, rkt_str, rkp_str, &pInvms_structs[iter.idx-1])) {
            yyjson_doc_free(pJson);
            free(an_str);
            free(rxss_str);
            free(rxrate_str);
            free(txp_str);
            free(rxlat_str);
            free(mtype_str);
            free(rkt_str);
            free(rkp_str);
            db_lab_free(&lab);
            for (int i = 0; i < iter.idx-1; i++) {
                db_invm_free(&pInvms_structs[i]);
            }
            free(pInvms_structs);
            return __lsapi_endpoint_error(pReq, 500, "Internal Server Error", "Failed to insert inventory measurement");
        }
        free(an_str);
        free(rxss_str);
        free(rxrate_str);
        free(txp_str);
        free(rxlat_str);
        free(mtype_str);
        free(rkt_str);
        free(rkp_str);
        db_lab_free(&lab);
    }

    LOG_V("__lsapi_endpoint_invm_bulk_put: iter.idx=%d, nInvms=%d", iter.idx, nInvms);
    if (iter.idx != nInvms) {
        yyjson_doc_free(pJson);
        for (int i = 0; i < iter.idx; i++) {
            db_invm_free(&pInvms_structs[i]);
        }
        free(pInvms_structs);
        return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Provided fewer inventory measurements than specified by n_invms");
    }

    static h2o_generator_t generator = {NULL, NULL};
    pReq->res.status = 200;
    pReq->res.reason = "OK";
    h2o_add_header(&pReq->pool, &pReq->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, H2O_STRLIT("application/json"));
    h2o_start_response(pReq, &generator);

    const char* status = "success";
    const char* message = "Inventory measurements inserted successfully (bulk)";
    
    // create json response
    yyjson_mut_doc* pJsonResp = yyjson_mut_doc_new(NULL);
    yyjson_mut_val* pRootResp = yyjson_mut_obj(pJsonResp);
    yyjson_mut_doc_set_root(pJsonResp, pRootResp);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "status", status);
    yyjson_mut_obj_add_str(pJsonResp, pRootResp, "message", message);
    // add inventory measurements data as an array in the json response
    yyjson_mut_val* pInvms_resp = yyjson_mut_arr(pJsonResp);
    for (int i = 0; i < nInvms; i++) {
        yyjson_mut_val* pInvm = yyjson_mut_obj(pJsonResp);
        yyjson_mut_obj_add_str(pJsonResp, pInvm, "t", pInvms_structs[i].time);
        yyjson_mut_obj_add_str(pJsonResp, pInvm, "epc", pInvms_structs[i].inventory_epc);
        yyjson_mut_obj_add_int(pJsonResp, pInvm, "an", pInvms_structs[i].antno);
        yyjson_mut_obj_add_int(pJsonResp, pInvm, "rx_signal_strength", pInvms_structs[i].rx_signal_strength);
        yyjson_mut_obj_add_int(pJsonResp, pInvm, "read_rate", pInvms_structs[i].read_rate);
        yyjson_mut_obj_add_int(pJsonResp, pInvm, "tx_power", pInvms_structs[i].tx_power);
        yyjson_mut_obj_add_int(pJsonResp, pInvm, "read_latency", pInvms_structs[i].read_latency);
        yyjson_mut_obj_add_int(pJsonResp, pInvm, "measurement_type", pInvms_structs[i].measurement_type);
        yyjson_mut_obj_add_int(pJsonResp, pInvm, "rotator_ktheta", pInvms_structs[i].rotator_ktheta);
        yyjson_mut_obj_add_int(pJsonResp, pInvm, "rotator_kphi", pInvms_structs[i].rotator_kphi);
        // add inventory measurement object to the array
        yyjson_mut_arr_add_val(pInvms_resp, pInvm);
    }
    yyjson_mut_obj_add_val(pJsonResp, pRootResp, "invms", pInvms_resp);

    char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
    assert(respText != NULL);
    h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
    h2o_send(pReq, &body, 1, 1);

    free((void*)respText);
    yyjson_doc_free(pJson);
    yyjson_mut_doc_free(pJsonResp);
    for (int i = 0; i < nInvms; i++) {
        db_invm_free(&pInvms_structs[i]);
    }
    free(pInvms_structs);
    return 0;
}

int lsapi_endpoint_invm_bulk(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);
    if (!h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("PUT"))) {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
    //return __lsapi_endpoint_invm_bulk_put(pH2oHandler, pReq, pLsapi);
    return __lsapi_endpoint_invm_bulk_put_optimized(pH2oHandler, pReq, pLsapi);
}
