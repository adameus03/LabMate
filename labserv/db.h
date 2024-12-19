#ifndef DB_H
#define DB_H

typedef struct db db_t;

typedef struct db_user {
  int user_id;
  char* passwd_hash;
  int role;
  char* ip_addr;
  char* registration_date;
  char* last_login_date;
  char* username;
  char* first_name;
  char* last_name;
  char* bio;
  int num_requests;
  int karma;
  char* email;
  int is_email_verified;
  char* email_verification_token_hash;
  char* sesskey_hash;
  char* last_usr_chng_date;
  char* sesskey_salt;
  char* passwd_salt;
  char* email_verification_token_salt;
} db_user_t;

/**
 * @brief Create a new database driver instance
 */
db_t* db_new();

/**
 * @brief Free resource allocation caused by `db_new`
 */
void db_free(db_t* pDb);

/**
 * @brief Initialize database driver
 */
void db_init(db_t* pDb);

/**
 * @brief Free resource allocation caused by `db_init`
 */
void db_close(db_t* pDb);

int db_user_insert_basic(db_t* pDb, 
                  const char* username,
                  const char* ip_addr,
                  const char* first_name,
                  const char* last_name,
                  const char* email,
                  const char* password_hash,
                  const char* password_salt,
                  const char* email_verification_token_hash,
                  const char* email_verification_token_salt);

int db_user_get_by_username(db_t* pDb, 
                            const char* username_in,
                            db_user_t* pUser_out);

int db_user_get_by_email(db_t* pDb, 
                         const char* email_in,
                         db_user_t* pUser_out);

int db_user_get_by_id(db_t* pDb, 
                      const char* user_id_in,
                      db_user_t* pUser_out);
                            

#endif // DB_H