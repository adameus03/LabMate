#include "db.h"
#include <postgresql/libpq-fe.h>
#include <stdlib.h>
#include <assert.h>
#include <plibsys/plibsys.h>
#include "log.h"
#include "config.h"

// struct db {
//   PGconn* pConn;
// };

// The connection pool will not shrink below DB_CONNECTION_POOL_INIT_SIZE
#define DB_CONNECTION_POOL_INIT_SIZE 8
#define DB_CONNECTION_POOL_MAX_SIZE 8
#define DB_CONNECTION_POOL_GROWTH_FACTOR 2
#define DB_CONNECTION_POOL_SHRINK_FACTOR 2
// If the pool is at 1/DB_CONNECTION_POOL_SHRINK_TRIGGER_FACTOR capacity, we shrink it
#define DB_CONNECTION_POOL_SHRINK_TRIGGER_FACTOR 4

typedef struct db_connection {
  PGconn* pConn;
  PMutex* pMutex;
  int conn_id;
} db_connection_t;

typedef struct db_connection_pool {
  db_connection_t* pConnections;
  int num_locked_connections; // Warning: Only for atomic use!
  int pool_size;
  /* Coarse lock for accessing pConnections and pool_size. A write lock is used when resizing the pool. Otherwise, a read lock shall be used */
  PRWLock* conn_rwlock;

} db_connection_pool_t;

struct db {
  db_connection_pool_t connection_pool;
};

// TODO Handle timeouts if needed?
static void __db_connection_init(db_connection_t* pDbConnection, int conn_id) {
  assert(pDbConnection != NULL);
  pDbConnection->pConn = PQconnectdb(LABSERV_DB_CONNINFO);
  if (CONNECTION_OK != PQstatus(pDbConnection->pConn)) {
    LOG_E("__db_connection_init: PostgreSql db connection error: %s", PQerrorMessage(pDbConnection->pConn));
    PQfinish(pDbConnection->pConn);
    exit(EXIT_FAILURE);
  }

  // Libpq docs say that this is needed to "set always-secure search path, so malicious users can't take control"
  PGresult* pResult = PQexec(pDbConnection->pConn, "SELECT pg_catalog.set_config('search_path', '', false)");
  if (PGRES_TUPLES_OK != PQresultStatus(pResult)) {
    LOG_E("__db_connection_init: search_path part 1 configuration failed: %s", PQerrorMessage(pDbConnection->pConn));
    PQclear(pResult);
    PQfinish(pDbConnection->pConn);
    exit(EXIT_FAILURE);
  }
  PQclear(pResult);

  // Set search path to include public and timescaledb schemas (it was really hard to guess why create_hypertable was not found in the timescaledb extension)
  pResult = PQexec(pDbConnection->pConn, "SELECT pg_catalog.set_config('search_path', 'public, timescaledb', false)");
  if (PGRES_TUPLES_OK != PQresultStatus(pResult)) {
    LOG_E("__db_connection_init: search_path part 2 configuration failed: %s", PQerrorMessage(pDbConnection->pConn));
    PQclear(pResult);
    PQfinish(pDbConnection->pConn);
    exit(EXIT_FAILURE);
  }
  PQclear(pResult);

  // Initialize mutex
  pDbConnection->pMutex = p_mutex_new();
  if (pDbConnection->pMutex == NULL) {
    LOG_E("__db_connection_init: Failed to allocate memory for mutex");
    exit(EXIT_FAILURE);
  }

  pDbConnection->conn_id = conn_id;
}

static void __db_connection_close(db_connection_t* pDbConnection) {
  assert(pDbConnection != NULL);
  PQfinish(pDbConnection->pConn);
  assert(pDbConnection->pMutex != NULL);
  p_mutex_free(pDbConnection->pMutex);
}

static void __db_connection_pool_init(db_connection_pool_t* pPool) {
  assert(pPool != NULL);

  pPool->pConnections = (db_connection_t*)malloc(DB_CONNECTION_POOL_INIT_SIZE * sizeof(db_connection_t));
  if (pPool->pConnections == NULL) {
    LOG_E("__db_connection_pool_init: Failed to allocate memory for connection pool");
    exit(EXIT_FAILURE);
  }
  
  pPool->pool_size = DB_CONNECTION_POOL_INIT_SIZE;
  for (int i = 0; i < pPool->pool_size; i++) {
    __db_connection_init(&pPool->pConnections[i], i);
  }

  pPool->num_locked_connections = 0;
  pPool->conn_rwlock = p_rwlock_new();
  if (pPool->conn_rwlock == NULL) {
    LOG_E("__db_connection_pool_init: Failed to create connection pool coarse rwlock");
    exit(EXIT_FAILURE);
  }
}

static void __db_connection_pool_close(db_connection_pool_t* pPool) {
  assert(pPool != NULL);
  for (int i = 0; i < pPool->pool_size; i++) {
    __db_connection_close(&pPool->pConnections[i]);
  }
  assert(pPool->pConnections != NULL);
  free(pPool->pConnections);

  assert(pPool->conn_rwlock != NULL);
  p_rwlock_free(pPool->conn_rwlock);
}

static db_connection_t* __db_connection_make_cpy(db_connection_t* pDbConnection) {
  assert(pDbConnection != NULL);
  db_connection_t* pDbConnectionCopy = (db_connection_t*)malloc(sizeof(db_connection_t));
  if (pDbConnectionCopy == NULL) {
    LOG_E("__db_connection_make_cpy: Failed to allocate memory for connection copy");
    exit(EXIT_FAILURE);
  }
  *pDbConnectionCopy = *pDbConnection;
  return pDbConnectionCopy;
}

static void  __db_connection_free_cpy(db_connection_t* pDbConnection) {
  assert(pDbConnection != NULL);
  free(pDbConnection);
}

// TODO (t20398470) Daemonize pool reallocations?
/**
 * @brief Obtain a connection from the pool. 
 * @note Handles pool expansion and connection locking [trylock, lock (if pool is full)]
 * @note We return a connection copy made with __db_connection_make_cpy and expect it to be freed by __db_connection_return_to_pool using __db_connection_free_cpy. We made this copy to prevent concurrency issues, BUT doesn't the write lock on pool reallocations (which waits for all connections to be unlocked) prevent the reallocations from messing up connection memory addresses? //TODO (t05329423432) remove the copying (__db_connection_make_cpy and __db_connection_free_cpy), because it is unnecessary (low priority - maybe it can be treated as defensive programming?)
 * @return A locked connection from the pool
 */
static db_connection_t* __db_connection_take_from_pool(db_connection_pool_t* pPool) {
  assert(TRUE == p_rwlock_reader_lock(pPool->conn_rwlock));
  assert(pPool != NULL);
  assert(pPool->pConnections != NULL);
  assert(pPool->pool_size >= DB_CONNECTION_POOL_INIT_SIZE);
  assert(pPool->pool_size <= DB_CONNECTION_POOL_MAX_SIZE);
  int num_locked_connections_atomic = p_atomic_int_get(&pPool->num_locked_connections);
  assert(num_locked_connections_atomic >= 0);
  //assert(num_locked_connections_atomic <= pPool->pool_size); //TODO reenable this when we fix consistency issues with `num_locked_connections_atomic`
  for (int i = 0; i < pPool->pool_size; i++) {
    assert(pPool->pConnections[i].pMutex != NULL);
    if (p_mutex_trylock(pPool->pConnections[i].pMutex)) {
      p_atomic_int_inc(&pPool->num_locked_connections);
      LOG_V("__db_connection_take_from_pool: Connection %d taken from pool. pPool->num_locked_connections = atomic: %d, actual: %d", i, num_locked_connections_atomic+1, p_atomic_int_get(&pPool->num_locked_connections));
      db_connection_t* pDbConnectionCopy = __db_connection_make_cpy(&pPool->pConnections[i]);
      assert(TRUE == p_rwlock_reader_unlock(pPool->conn_rwlock));
      return pDbConnectionCopy;
    }
  }
    
  if (pPool->pool_size == DB_CONNECTION_POOL_MAX_SIZE) {
    LOG_W("__db_connection_take_from_pool: Connection pool already at max size. Waiting for pConnections[0].pMutex to become available");
    assert(pPool->pConnections[0].pMutex != NULL);
    assert(TRUE == p_mutex_lock(pPool->pConnections[0].pMutex));
    p_atomic_int_inc(&pPool->num_locked_connections);
    LOG_V("__db_connection_take_from_pool: Connection 0 taken from pool. pPool->num_locked_connections = atomic: %d, actual: %d", num_locked_connections_atomic+1, p_atomic_int_get(&pPool->num_locked_connections));
    db_connection_t* pDbConnectionCopy = __db_connection_make_cpy(&pPool->pConnections[0]);
    assert(TRUE == p_rwlock_reader_unlock(pPool->conn_rwlock));
    return pDbConnectionCopy;
  }

  //If we got here, then all connections are locked, but we can still grow the pool
  assert(pPool->pool_size < DB_CONNECTION_POOL_MAX_SIZE);

  int old_pool_size = pPool->pool_size;
  assert(TRUE == p_rwlock_reader_unlock(pPool->conn_rwlock));
  assert(TRUE == p_rwlock_writer_lock(pPool->conn_rwlock));
  if (old_pool_size != pPool->pool_size) {
    LOG_W("__db_connection_take_from_pool: Pool size changed from %d to %d while waiting for write lock. Releasing lock and reentering __db_connection_take_from_pool", old_pool_size, pPool->pool_size);
    assert(TRUE == p_rwlock_writer_unlock(pPool->conn_rwlock));
    return __db_connection_take_from_pool(pPool);
  }

  //If we got here then we have the write lock and the pool size is still the same so we can safely expand the pool
  int new_pool_size = DB_CONNECTION_POOL_GROWTH_FACTOR * pPool->pool_size;
  if (new_pool_size > DB_CONNECTION_POOL_MAX_SIZE) {
    new_pool_size = DB_CONNECTION_POOL_MAX_SIZE;
    LOG_I("__db_connection_take_from_pool: Expanding connection pool from size %d to %d", pPool->pool_size, new_pool_size);
    LOG_W("__db_connection_take_from_pool: Connection pool size limit reached. Reallocating and stopping at pool size %d", new_pool_size);
  } else {
    LOG_I("__db_connection_take_from_pool: Expanding connection pool from size %d to %d", pPool->pool_size, new_pool_size);
  }
  db_connection_t* pNewConnections = (db_connection_t*)realloc(pPool->pConnections, DB_CONNECTION_POOL_GROWTH_FACTOR * pPool->pool_size * sizeof(db_connection_t));
  if (pNewConnections == NULL) {
    LOG_E("__db_connection_take_from_pool: Failed to reallocate memory for connection pool");
    exit(EXIT_FAILURE);
  }
  pPool->pConnections = pNewConnections;
  for (int i = pPool->pool_size; i < new_pool_size; i++) {
    __db_connection_init(&pPool->pConnections[i], i);
  }
  pPool->pool_size = new_pool_size;
  p_atomic_int_inc(&pPool->num_locked_connections);
  LOG_V("__db_connection_take_from_pool: Connection %d taken from pool. pPool->num_locked_connections = atomic: %d, actual: %d", old_pool_size, num_locked_connections_atomic+1, p_atomic_int_get(&pPool->num_locked_connections));
  db_connection_t* pDbConnectionCopy = __db_connection_make_cpy(&pPool->pConnections[old_pool_size]);
  assert(TRUE == p_rwlock_writer_unlock(pPool->conn_rwlock));
  return pDbConnectionCopy;
}


/**
 * @note See @t20398470 for a note on pool reallocations
 * @note In pool shrinking we do coarse writer lock of course
 * @note The locked connections in the pool can occupy any position - we check if the rightmost connections are unlocked before shrinking. If they are not, we give up on shrinking instead of recursively trying to shrink the pool (could it be bad?). 
 * @note (n240935) Relocating the connections to the leftmost positions could possibly be done, but it would break the meaning of `conn_id` which is used so that db_connection_t copy can be made in db_connection_take_from_pool (we don't want to return the original because pool realocations would cause a neccessity to perform a search for the connection in the pool when returning it - that's why we want `conn_id` to have a correct value - updating it would possibly come with concurrency issues)
 * @note (n23049242) Update to note n240935: While we have the write lock we can safely update the conn_id of the connections in the pool, because we have the write lock and no other thread can access the pool at the same time. So the connection relocation (don't confuse with reallocation) could be implemented, however would it increase or decrease efficiency? Possible //TODO (see n230293420343 also - may not be a good idea possibly)
 * @note (n230293420343) Update to note n23049242: The connections which would be relocated would probably be corresponding to long-executing database queries. In order to obtain write lock for the pool, we would need to wait for it to complete (coarse rwlock unlocked). So the relocation would probably not be a good idea.
 */
static void __db_connection_return_to_pool(db_connection_t* pConn, db_connection_pool_t* pPool) {
  assert(TRUE == p_rwlock_reader_lock(pPool->conn_rwlock));
  assert(pConn != NULL);
  assert(pPool != NULL);
  assert(pPool->pConnections != NULL);
  assert(pPool->pool_size >= DB_CONNECTION_POOL_INIT_SIZE);
  assert(pPool->pool_size <= DB_CONNECTION_POOL_MAX_SIZE);
  //assert(pConn >= pPool->pConnections);
  //assert(pConn < pPool->pConnections + pPool->pool_size);
  assert(!(pConn >= pPool->pConnections && pConn < pPool->pConnections + pPool->pool_size)); // Make sure if pConn is not the original structure (it should be a copy made with __db_connection_make_cpy) //TODO remove this and replace with reverse check (like previously - see 2 commented out assert lines above if they're still there) if t05329423432 is done
  assert(pConn->conn_id >= 0);
  LOG_V("__db_connection_return_to_pool: pConn->conn_id = %d, pPool->pool_size = %d", pConn->conn_id, pPool->pool_size);
  assert(pConn->conn_id < pPool->pool_size);
  assert(pConn->pMutex != NULL);
  int num_locked_connections_atomic = p_atomic_int_get(&pPool->num_locked_connections);
  assert(num_locked_connections_atomic >= 0);
  //assert(num_locked_connections_atomic <= pPool->pool_size); //TODO reenable this when we fix consistency issues with `num_locked_connections_atomic`

  assert(TRUE == p_mutex_unlock(pConn->pMutex));
  p_atomic_int_dec_and_test(&pPool->num_locked_connections);
  //db_connection_t* __relPtr = (db_connection_t*)(pConn - pPool->pConnections);
  LOG_V("__db_connection_return_to_pool: Connection %d returned to pool. pPool->num_locked_connections = atomic: %d, actual: %d", pConn->conn_id, num_locked_connections_atomic-1, p_atomic_int_get(&pPool->num_locked_connections));
  __db_connection_free_cpy(pConn); 
  pConn = NULL; // We won't use it anymore in this function

  // If the pool is at 1/DB_CONNECTION_POOL_SHRINK_TRIGGER_FACTOR capacity, we shrink it
  if ((num_locked_connections_atomic-1) * DB_CONNECTION_POOL_SHRINK_TRIGGER_FACTOR <= pPool->pool_size) {
    if (pPool->pool_size == DB_CONNECTION_POOL_INIT_SIZE) {
      LOG_I("__db_connection_return_to_pool: Pool already at minimum size, not shrinking below initial size %d", DB_CONNECTION_POOL_INIT_SIZE);
      assert(TRUE == p_rwlock_reader_unlock(pPool->conn_rwlock));
      return;
    }
    int new_pool_size = pPool->pool_size / DB_CONNECTION_POOL_SHRINK_FACTOR;
    if (new_pool_size < DB_CONNECTION_POOL_INIT_SIZE) {
      new_pool_size = DB_CONNECTION_POOL_INIT_SIZE;
    }
    if (new_pool_size == pPool->pool_size) {
      LOG_W("__db_connection_return_to_pool: Shrinked pool size is the same as current pool size! Not reallocating");
      assert(TRUE == p_rwlock_reader_unlock(pPool->conn_rwlock));
      return;
    }

    // Check if the rightmost connections are unlocked
    for (int i = pPool->pool_size - 1; i >= new_pool_size; i--) {
      if (FALSE == p_mutex_trylock(pPool->pConnections[i].pMutex)) {
        LOG_W("__db_connection_return_to_pool: Connection %d is still locked. Not shrinking pool", i);
        // Unlock the connections that we locked so far while checking
        for (int j = pPool->pool_size - 1; j > i; j--) {
          assert(TRUE == p_mutex_unlock(pPool->pConnections[j].pMutex));
        }
        assert(TRUE == p_rwlock_reader_unlock(pPool->conn_rwlock));
        return;
      }
    }

    // We know `__db_connection_return_to_pool` won't shrink the pool concurrently, because we have the reader lock
    // However a pool expansion can happen, that's why we need to compare the pool size before and after upgrading the lock
    // If the pool size changed, we unlock the connections which we locked for closing and give up on shrinking
    int old_pool_size = pPool->pool_size;
    assert(TRUE == p_rwlock_reader_unlock(pPool->conn_rwlock));
    assert(TRUE == p_rwlock_writer_lock(pPool->conn_rwlock));
    if (old_pool_size != pPool->pool_size) {
      LOG_W("__db_connection_return_to_pool: Pool size changed from %d to %d while waiting for write lock. Releasing lock and giving up on pool shrinking", old_pool_size, pPool->pool_size);
      for (int i = pPool->pool_size - 1; i >= new_pool_size; i--) {
        assert(TRUE == p_mutex_unlock(pPool->pConnections[i].pMutex));
      }
      assert(TRUE == p_rwlock_writer_unlock(pPool->conn_rwlock));
      return;
    }
    //LOG_W("__db_connection_return_to_pool: Abort shrinking FOR NOW"); for (int i = new_pool_size; i < pPool->pool_size; i++) {  assert(TRUE == p_mutex_unlock(pPool->pConnections[i].pMutex)); } assert(TRUE == p_rwlock_writer_unlock(pPool->conn_rwlock)); return; //TODO remove this line when segfault bug is fixed
    // If we got here then we have the write lock and the pool size is still the same so we can safely shrink the pool
    LOG_I("__db_connection_return_to_pool: Shrinking connection pool from size %d to %d", pPool->pool_size, new_pool_size);
    for (int i = new_pool_size; i < pPool->pool_size; i++) {
      assert(TRUE == p_mutex_unlock(pPool->pConnections[i].pMutex));
      __db_connection_close(&pPool->pConnections[i]);
    }
    db_connection_t* pNewConnections = (db_connection_t*)realloc(pPool->pConnections, new_pool_size * sizeof(db_connection_t));
    if (pNewConnections == NULL) {
      LOG_F("__db_connection_return_to_pool: Failed to reallocate memory for shrinked connection pool");
      exit(EXIT_FAILURE);
    }
    pPool->pConnections = pNewConnections;
    pPool->pool_size = new_pool_size;
    assert(TRUE == p_rwlock_writer_unlock(pPool->conn_rwlock));//here I think we are done
  } else {
    assert(TRUE == p_rwlock_reader_unlock(pPool->conn_rwlock));
  }
}

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
  assert(pDb != NULL);
  int libpq_version = PQlibVersion();
  LOG_I("db_init: Libpq version: %d", libpq_version);
  int libpq_is_thread_safe = PQisthreadsafe();
  if (1 != libpq_is_thread_safe) {
    // Libpq is said to be "thread-safe" if PQisthreadsafe returns 1
    // This "thread safety" however, does not encompass using same PGconn in multiple threads (see libpq docs)
    LOG_E("db_init: Libpq is not \"thread-safe\" - PQisthreadsafe returned %d (expected 1). Exiting.", libpq_is_thread_safe);
    exit(EXIT_FAILURE);
  }
  
  __db_connection_pool_init(&pDb->connection_pool);
  db_connection_t* pDbConnection = __db_connection_take_from_pool(&pDb->connection_pool);

  if (0 != __db_exec_script(pDbConnection->pConn, LABSERV_DB_INIT_SCRIPT_PATH)) {
    LOG_E("db_init: Failed to execute init script");
    __db_connection_return_to_pool(pDbConnection, &pDb->connection_pool);
    __db_connection_pool_close(&pDb->connection_pool);
    exit(EXIT_FAILURE);
  }

  __db_connection_return_to_pool(pDbConnection, &pDb->connection_pool);
}

void db_close(db_t* pDb) {
  assert(pDb != NULL);
  __db_connection_pool_close(&pDb->connection_pool);
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
  
  db_connection_t* pDbConnection = __db_connection_take_from_pool(&pDb->connection_pool);
  PGresult* pResult = PQexecParams(pDbConnection->pConn, pQuery, 9, NULL, pParams, NULL, NULL, 0);
  if (PGRES_COMMAND_OK != PQresultStatus(pResult)) {
    LOG_E("db_user_insert: Failed to insert user (username \"%s\"): %s", username, PQerrorMessage(pDbConnection->pConn));
    PQclear(pResult);
    __db_connection_return_to_pool(pDbConnection, &pDb->connection_pool);
    return -1;
  }

  PQclear(pResult);
  __db_connection_return_to_pool(pDbConnection, &pDb->connection_pool);
  return 0;
}

/**
 * @brief Deep clone a db_user_t structure
 */
static db_user_t db_user_clone(db_user_t dbUser) {
  return (db_user_t) {
    .user_id = dbUser.user_id,
    .passwd_hash = p_strdup(dbUser.passwd_hash),
    .role = dbUser.role,
    .ip_addr = p_strdup(dbUser.ip_addr),
    .registration_date = p_strdup(dbUser.registration_date),
    .last_login_date = p_strdup(dbUser.last_login_date),
    .username = p_strdup(dbUser.username),
    .first_name = p_strdup(dbUser.first_name),
    .last_name = p_strdup(dbUser.last_name),
    .bio = p_strdup(dbUser.bio),
    .num_requests = dbUser.num_requests,
    .karma = dbUser.karma,
    .email = p_strdup(dbUser.email),
    .is_email_verified = dbUser.is_email_verified,
    .email_verification_token_hash = p_strdup(dbUser.email_verification_token_hash),
    .sesskey_hash = p_strdup(dbUser.sesskey_hash),
    .last_usr_chng_date = p_strdup(dbUser.last_usr_chng_date),
    .sesskey_salt = p_strdup(dbUser.sesskey_salt),
    .passwd_salt = p_strdup(dbUser.passwd_salt),
    .email_verification_token_salt = p_strdup(dbUser.email_verification_token_salt)
  };
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

  db_connection_t* pDbConnection = __db_connection_take_from_pool(&pDb->connection_pool);
  PGresult* pResult = PQexecParams(pDbConnection->pConn, pQuery, 1, NULL, pParams, NULL, NULL, 0);
  if (PGRES_TUPLES_OK != PQresultStatus(pResult)) {
    LOG_E("db_user_get_by_username: Failed to get user: %s", PQerrorMessage(pDbConnection->pConn));
    PQclear(pResult);
    __db_connection_return_to_pool(pDbConnection, &pDb->connection_pool);
    return -1;
  }

  if (PQntuples(pResult) == 0) {
    LOG_I("db_user_get_by_username: No user found");
    PQclear(pResult);
    __db_connection_return_to_pool(pDbConnection, &pDb->connection_pool);
    return -2; // No user found
  }
  if (PQntuples(pResult) > 1) {
    LOG_I("db_user_get_by_username: Multiple users found");
    PQclear(pResult);
    __db_connection_return_to_pool(pDbConnection, &pDb->connection_pool);
    return -3; // Multiple users found
  }
  if (PQnfields(pResult) != 20) {
    LOG_E("db_user_get_by_username: Unexpected number of fields in result: %d", PQnfields(pResult));
    PQclear(pResult);
    __db_connection_return_to_pool(pDbConnection, &pDb->connection_pool);
    exit(EXIT_FAILURE);
  }

  db_user_t user;
  user.user_id = atoi(PQgetvalue(pResult, 0, 0));
  user.passwd_hash = PQgetvalue(pResult, 0, 1);
  user.role = atoi(PQgetvalue(pResult, 0, 2));
  user.ip_addr = PQgetvalue(pResult, 0, 3);
  user.registration_date = PQgetvalue(pResult, 0, 4);
  user.last_login_date = PQgetvalue(pResult, 0, 5);
  user.username = PQgetvalue(pResult, 0, 6);
  user.first_name = PQgetvalue(pResult, 0, 7);
  user.last_name = PQgetvalue(pResult, 0, 8);
  user.bio = PQgetvalue(pResult, 0, 9);
  user.num_requests = atoi(PQgetvalue(pResult, 0, 10));
  user.karma = atoi(PQgetvalue(pResult, 0, 11));
  user.email = PQgetvalue(pResult, 0, 12);
  user.is_email_verified = PQgetvalue(pResult, 0, 13)[0] == 't' ? TRUE : FALSE;
  user.email_verification_token_hash = PQgetvalue(pResult, 0, 14);
  user.sesskey_hash = PQgetvalue(pResult, 0, 15);
  user.last_usr_chng_date = PQgetvalue(pResult, 0, 16);
  user.sesskey_salt = PQgetvalue(pResult, 0, 17);
  user.passwd_salt = PQgetvalue(pResult, 0, 18);
  user.email_verification_token_salt = PQgetvalue(pResult, 0, 19);
  
  *pUser_out = db_user_clone(user);

  PQclear(pResult);
  __db_connection_return_to_pool(pDbConnection, &pDb->connection_pool);
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

int db_user_set_email_verified(db_t* pDb, const char* username) {
  assert(pDb != NULL);
  assert(username != NULL);
  const char* pQuery = "UPDATE public.users SET email_verified = true WHERE username = $1";
  const char* pParams[1] = {username};
  db_connection_t* pDbConnection = __db_connection_take_from_pool(&pDb->connection_pool);
  PGresult* pResult = PQexecParams(pDbConnection->pConn, pQuery, 1, NULL, pParams, NULL, NULL, 0);
  if (PGRES_COMMAND_OK != PQresultStatus(pResult)) {
    LOG_E("db_user_set_email_verified: Failed to set email_verified for username \"%s\": %s", username, PQerrorMessage(pDbConnection->pConn));
    PQclear(pResult);
    __db_connection_return_to_pool(pDbConnection, &pDb->connection_pool);
    return -1;
  }
  PQclear(pResult);
  __db_connection_return_to_pool(pDbConnection, &pDb->connection_pool);
  return 0;
}

int db_user_set_session(db_t* pDb, const char* username, const char* sesskey_hash, const char* sesskey_salt) {
  assert(pDb != NULL);
  assert(username != NULL);
  assert(sesskey_hash != NULL);
  assert(sesskey_salt != NULL);
  const char* pQuery = "UPDATE public.users SET sesskey_hash = $1, sesskey_salt = $2 WHERE username = $3";
  const char* pParams[3] = {sesskey_hash, sesskey_salt, username};
  db_connection_t* pDbConnection = __db_connection_take_from_pool(&pDb->connection_pool);
  PGresult* pResult = PQexecParams(pDbConnection->pConn, pQuery, 3, NULL, pParams, NULL, NULL, 0);
  if (PGRES_COMMAND_OK != PQresultStatus(pResult)) {
    LOG_E("db_user_set_session: Failed to set session for username \"%s\": %s", username, PQerrorMessage(pDbConnection->pConn));
    PQclear(pResult);
    __db_connection_return_to_pool(pDbConnection, &pDb->connection_pool);
    return -1;
  }
  PQclear(pResult);
  __db_connection_return_to_pool(pDbConnection, &pDb->connection_pool);
  return 0;
}

int db_user_unset_session(db_t* pDb, const char* username) {
  assert(pDb != NULL);
  assert(username != NULL);
  const char* pQuery = "UPDATE public.users SET sesskey_hash = NULL, sesskey_salt = NULL WHERE username = $1";
  const char* pParams[1] = {username};
  db_connection_t* pDbConnection = __db_connection_take_from_pool(&pDb->connection_pool);
  PGresult* pResult = PQexecParams(pDbConnection->pConn, pQuery, 1, NULL, pParams, NULL, NULL, 0);
  if (PGRES_COMMAND_OK != PQresultStatus(pResult)) {
    LOG_E("db_user_unset_session: Failed to unset session for username \"%s\": %s", username, PQerrorMessage(pDbConnection->pConn));
    PQclear(pResult);
    __db_connection_return_to_pool(pDbConnection, &pDb->connection_pool);
    return -1;
  }
  PQclear(pResult);
  __db_connection_return_to_pool(pDbConnection, &pDb->connection_pool);
  return 0;
}