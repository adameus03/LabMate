#include "db.h"
#include <postgresql/libpq-fe.h>
#include <stdlib.h>
#include "log.h"
#include "config.h"

struct db {
  PGconn* pConn;
};

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
}