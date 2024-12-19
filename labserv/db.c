#include "db.h"
#include <postgresql/libpq-fe.h>
#include <stdlib.h>
#include <assert.h>
#include "log.h"
#include "config.h"

struct db {
  PGconn* pConn;
};

static const char* __db_load_script(const char* script_path) {
  FILE* pFile = fopen(script_path, "r");
  if (pFile == NULL) {
    LOG_E("__db_load_script: Failed to open file %s", script_path);
    return NULL;
  }

  fseek(pFile, 0, SEEK_END);
  long lSize = ftell(pFile);
  rewind(pFile);

  char* pScript = (char*)malloc(lSize + 1);
  if (pScript == NULL) {
    LOG_E("__db_load_script: Failed to allocate memory for script");
    fclose(pFile);
    return NULL;
  }

  size_t result = fread(pScript, 1, lSize, pFile);
  if (result != lSize) {
    LOG_E("__db_load_script: Failed to read file %s", script_path);
    fclose(pFile);
    free(pScript);
    return NULL;
  }

  pScript[lSize] = '\0';
  fclose(pFile);
  return pScript;
}

static int __db_exec_script(PGconn* pConn, const char* script_path) {
  assert(pConn != NULL);
  assert(script_path != NULL);
  const char* pScriptText = __db_load_script(script_path);
  if (pScriptText == NULL) {
    return -1;
  }

  PGresult* pResult = PQexec(pConn, pScriptText);
  if (PGRES_COMMAND_OK != PQresultStatus(pResult)) {
    LOG_E("__db_exec_script: Failed to execute script %s: %s", script_path, PQerrorMessage(pConn));
    PQclear(pResult);
    free((void*)pScriptText);
    return -1;
  }

  PQclear(pResult);
  free((void*)pScriptText);
  return 0;
}


void db_init(db_t* pDb) {
  pDb->pConn = PQconnectdb(LABSERV_DB_CONNINFO);
  if (CONNECTION_OK != PQstatus(pDb->pConn)) {
    LOG_E("db_init: PostgreSql db connection error: %s", PQerrorMessage(pDb->pConn));
    PQfinish(pDb->pConn);
    exit(EXIT_FAILURE);
  }

  // Libpq docs say that this is needed to "set always-secure search path, so malicious users can't take control"
  PGresult* pResult = PQexec(pDb->pConn, "SELECT pg_catalog.set_config('search_path', '', false)");
  if (PGRES_TUPLES_OK != PQresultStatus(pResult)) {
    LOG_E("db_init: search_path configuration failed: %s", PQerrorMessage(pDb->pConn));
    PQclear(pResult);
    PQfinish(pDb->pConn);
    exit(EXIT_FAILURE);
  }
  PQclear(pResult);

  if (0 != __db_exec_script(pDb->pConn, LABSERV_DB_INIT_SCRIPT_PATH)) {
    PQfinish(pDb->pConn);
    exit(EXIT_FAILURE);
  }
}

void db_close(db_t* pDb) {
  assert(pDb != NULL);
  PQfinish(pDb->pConn);
}

db_t* db_new() {
  db_t* pDb = (db_t*)malloc(sizeof(db_t));
  if (pDb == NULL) {
    LOG_E("db_new: Failed to allocate memory for db_t");
    exit(EXIT_FAILURE);
  }

  return pDb;
}

void db_free(db_t* pDb) {
  assert(pDb != NULL);
  free(pDb);
}

int db_user_insert_basic(db_t* pDb, 
                  const char* username,
                  const char* ip_addr,
                  const char* first_name,
                  const char* last_name,
                  const char* email,
                  const char* password_hash,
                  const char* password_salt,
                  const char* email_verification_token_hash,
                  const char* email_verification_token_salt) {
  assert(pDb != NULL);
  assert(username != NULL);
  assert(ip_addr != NULL);
  assert(first_name != NULL);
  assert(last_name != NULL);
  assert(email != NULL);
  assert(password_hash != NULL);
  assert(password_salt != NULL);
  assert(email_verification_token_hash != NULL);
  assert(email_verification_token_salt != NULL);
  
  const char* pQuery = "INSERT INTO public.users (username, ip_addr, first_name, last_name, email, passwd_hash, email_verification_token_hash, email_verification_token_salt, passwd_salt, registration_date) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())";
  const char* pParams[9] = {username, ip_addr, first_name, last_name, email, password_hash, email_verification_token_hash, email_verification_token_salt, password_salt};
  PGresult* pResult = PQexecParams(pDb->pConn, pQuery, 9, NULL, pParams, NULL, NULL, 0);
  if (PGRES_COMMAND_OK != PQresultStatus(pResult)) {
    LOG_E("db_user_insert: Failed to insert user: %s", PQerrorMessage(pDb->pConn));
    PQclear(pResult);
    return -1;
  }

  PQclear(pResult);
  return 0;
}

static int db_user_get_by_x(db_t* pDb,
                            const char* pQuery,
                            const char** pParams,
                            int nParams,
                            db_user_t* pUser_out) {
  assert(pDb != NULL);
  assert(pQuery != NULL);
  assert(pParams != NULL);
  assert(nParams > 0);
  assert(pUser_out != NULL);
  PGresult* pResult = PQexecParams(pDb->pConn, pQuery, 1, NULL, pParams, NULL, NULL, 0);
  if (PGRES_TUPLES_OK != PQresultStatus(pResult)) {
    LOG_E("db_user_get_by_username: Failed to get user: %s", PQerrorMessage(pDb->pConn));
    PQclear(pResult);
    return -1;
  }

  if (PQntuples(pResult) == 0) {
    LOG_I("db_user_get_by_username: No user found");
    PQclear(pResult);
    return -2; // No user found
  }
  if (PQntuples(pResult) > 1) {
    LOG_I("db_user_get_by_username: Multiple users found");
    PQclear(pResult);
    return -3; // Multiple users found
  }
  if (PQnfields(pResult) != 20) {
    LOG_E("db_user_get_by_username: Unexpected number of fields in result: %d", PQnfields(pResult));
    PQclear(pResult);
    exit(EXIT_FAILURE);
  }

  pUser_out->user_id = atoi(PQgetvalue(pResult, 0, 0));
  pUser_out->passwd_hash = PQgetvalue(pResult, 0, 1);
  pUser_out->role = atoi(PQgetvalue(pResult, 0, 2));
  pUser_out->ip_addr = PQgetvalue(pResult, 0, 3);
  pUser_out->registration_date = PQgetvalue(pResult, 0, 4);
  pUser_out->last_login_date = PQgetvalue(pResult, 0, 5);
  pUser_out->username = PQgetvalue(pResult, 0, 6);
  pUser_out->first_name = PQgetvalue(pResult, 0, 7);
  pUser_out->last_name = PQgetvalue(pResult, 0, 8);
  pUser_out->bio = PQgetvalue(pResult, 0, 9);
  pUser_out->num_requests = atoi(PQgetvalue(pResult, 0, 10));
  pUser_out->karma = atoi(PQgetvalue(pResult, 0, 11));
  pUser_out->email = PQgetvalue(pResult, 0, 12);
  pUser_out->is_email_verified = atoi(PQgetvalue(pResult, 0, 13));
  pUser_out->email_verification_token_hash = PQgetvalue(pResult, 0, 14);
  pUser_out->sesskey_hash = PQgetvalue(pResult, 0, 15);
  pUser_out->last_usr_chng_date = PQgetvalue(pResult, 0, 16);
  pUser_out->sesskey_salt = PQgetvalue(pResult, 0, 17);
  pUser_out->passwd_salt = PQgetvalue(pResult, 0, 18);
  pUser_out->email_verification_token_salt = PQgetvalue(pResult, 0, 19);

  PQclear(pResult);
  return 0; // Success
}

int db_user_get_by_username(db_t* pDb, 
                            const char* username_in,
                            db_user_t* pUser_out) {
  assert(pDb != NULL);
  assert(username_in != NULL);
  assert(pUser_out != NULL);
  const char* pQuery = "SELECT * FROM public.users WHERE username = $1";
  const char* pParams[1] = {username_in};
  return db_user_get_by_x(pDb, pQuery, pParams, 1, pUser_out);
}

int db_user_get_by_email(db_t* pDb, 
                         const char* email_in,
                         db_user_t* pUser_out) {
  assert(pDb != NULL);
  assert(email_in != NULL);
  assert(pUser_out != NULL);
  const char* pQuery = "SELECT * FROM public.users WHERE email = $1";
  const char* pParams[1] = {email_in};
  return db_user_get_by_x(pDb, pQuery, pParams, 1, pUser_out);
}

int db_user_get_by_id(db_t* pDb, 
                      const char* user_id_in,
                      db_user_t* pUser_out) {
  assert(pDb != NULL);
  assert(user_id_in > 0);
  assert(pUser_out != NULL);
  const char* pQuery = "SELECT * FROM public.users WHERE user_id = $1";
  
  const char* pParams[1] = {user_id_in};
  return db_user_get_by_x(pDb, pQuery, pParams, 1, pUser_out);
}