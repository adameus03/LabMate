#include "lsapi.h"
#include <yyjson.h>
#include <bcrypt/bcrypt.h>
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
    size_t result = fread(token, 1, 32, f);
    if (result != __LSAPI_EMAIL_VERIF_TOKEN_LEN) {
        LOG_E("__lsapi_generate_token: Failure when reading /dev/urandom (fread returned %lu while expecting %d)", result, __LSAPI_EMAIL_VERIF_TOKEN_LEN);
        free(token);
        fclose(f);
        return NULL;
    }
    fclose(f);
    for (int i = 0; i < __LSAPI_EMAIL_VERIF_TOKEN_LEN; i++) {
        token[i] = 'a' + (token[i] % 26);
    }
    return token;
}

static void lsapi_email_send(char* verification_token) {
    assert(verification_token != NULL);
    LOG_I("lsapi_email_handler: Dummy email handler");
    char* token = malloc(strlen(verification_token) + 1);
    if (token == NULL) {
        LOG_E("lsapi_email_handler: Failed to allocate memory for token");
        exit(EXIT_FAILURE);
    }
    strcpy(token, verification_token);
    // TODO spawn a new thread to send email
    free(token);
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
    pReq->res.status = httpStatus;
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

// curl -X POST -d '{"username":"abc","email":"abc@example.com","password":"test","first_name":"test","last_name":"test"}' http://localhost:7890/api/user
int lsapi_endpoint_user(h2o_handler_t* pH2oHandler, h2o_req_t* pReq)
{
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);
    lsapi_t* pLsapi = __lsapi_self_from_h2o_handler(pH2oHandler);
    if (h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("POST"))) {
        yyjson_doc* pJson = yyjson_read(pReq->entity.base, pReq->entity.len, 0);
        if (pJson == NULL) {
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Invalid JSON");
        } 
        yyjson_val* pRoot = yyjson_doc_get_root(pJson);
        if (pRoot == NULL || !yyjson_is_obj(pRoot)) {
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing JSON root object");
        }
        yyjson_val* pUsername = yyjson_obj_get(pRoot, "username");
        if (pUsername == NULL || !yyjson_is_str(pUsername)) {
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid username");
        }
        yyjson_val* pEmail = yyjson_obj_get(pRoot, "email");
        if (pEmail == NULL || !yyjson_is_str(pEmail)) {
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid email");
        }
        yyjson_val* pPassword = yyjson_obj_get(pRoot, "password");
        if (pPassword == NULL || !yyjson_is_str(pPassword)) {
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid password");
        }
        yyjson_val* pFirstName = yyjson_obj_get(pRoot, "first_name");
        if (pFirstName == NULL || !yyjson_is_str(pFirstName)) {
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid first name");
        }
        yyjson_val* pLastName = yyjson_obj_get(pRoot, "last_name");
        if (pLastName == NULL || !yyjson_is_str(pLastName)) {
            return __lsapi_endpoint_error(pReq, 400, "Bad Request", "Missing or invalid last name");
        }
        
        const char* username = yyjson_get_str(pUsername);
        const char* first_name = yyjson_get_str(pFirstName);
        const char* last_name = yyjson_get_str(pLastName);
        const char* email = yyjson_get_str(pEmail);
        const char* password = yyjson_get_str(pPassword);
        
        assert(username != NULL && first_name != NULL && last_name != NULL && email != NULL && password != NULL);
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
            pReq->res.status = 500;
            pReq->res.reason = "Internal Server Error";
            h2o_send_inline(pReq, H2O_STRLIT("Internal Server Error"));
            yyjson_doc_free(pJson);
            return 0;
        }

        //hash email verification token
        char email_verif_token_hash[BCRYPT_HASHSIZE];
        char email_verif_token_salt[BCRYPT_HASHSIZE];
        assert(0 == bcrypt_gensalt(12, email_verif_token_salt));
        assert(0 == bcrypt_hashpw(email_verif_token, email_verif_token_salt, email_verif_token_hash));

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
            return __lsapi_endpoint_error(pReq, 409, "Conflict", "Username/email already exists");
        }

        db_user_t user;
        if (0 != db_user_get_by_username(pLsapi->pDb, username, &user)) {
            pReq->res.status = 500; // Internal Server Error
            pReq->res.reason = "Internal Server Error";
            h2o_send_inline(pReq, H2O_STRLIT("Internal Server Error"));
            yyjson_doc_free(pJson);
            return 0;
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
        
        const char* respText = yyjson_mut_write(pJsonResp, 0, NULL);
        assert(respText != NULL);
        h2o_iovec_t body = h2o_strdup(&pReq->pool, respText, SIZE_MAX);
        h2o_send(pReq, &body, 1, 1);

        free((void*)respText);
        yyjson_doc_free(pJson);
        yyjson_mut_doc_free(pJsonResp);

        lsapi_email_send(email_verif_token);
        free(email_verif_token);
        return 0;
    } else {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }
}

int lsapi_endpoint_service_status(h2o_handler_t* pH2oHandler, h2o_req_t* pReq) {
    assert(pH2oHandler != NULL);
    assert(pReq != NULL);

    if (!h2o_memis(pReq->method.base, pReq->method.len, H2O_STRLIT("GET"))) {
        return __lsapi_endpoint_error(pReq, 405, "Method Not Allowed", "Method Not Allowed");
    }

    return __lsapi_endpoint_success(pReq, 200, "OK", "Service is running");
}