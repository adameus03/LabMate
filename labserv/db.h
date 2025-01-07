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

typedef struct db_faculty {
  int faculty_id;
  char* name;
  char* email_domain;
} db_faculty_t;

typedef struct db_reagent_type {
  int reagtype_id;
  char* name;
} db_reagent_type_t;

typedef struct db_lab {
  int lab_id;
  char* name;
  char* bearer_token_hash;
  char* bearer_token_salt;
  int faculty_id;
} db_lab_t;

typedef struct db_reagent {
  int reagent_id;
  char* name;
  int reagent_type_id;
  char* vendor;
} db_reagent_t;

typedef struct db_inventory_item {
  int inventory_id;
  int reagent_id;
  char* date_added;
  char* date_expire;
  int lab_id;
  char* epc;
} db_inventory_item_t;

typedef struct db_antenna {
  int antenna_id;
  char* name;
  char* info;
  int k;
  int lab_id;
} db_antenna_t;

typedef struct db_invm {
  char* time;
  char* inventory_epc;
  int antenna_id;
  int rx_signal_strength;
  int read_rate;
  int tx_power;
  int read_latency;
  int measurement_type;
  int rotator_ktheta; // For future when antennas can rotate
  int rotator_kphi; // For future when antennas can rotate
} db_invm_t;
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

int db_user_set_email_verified(db_t* pDb, const char* username);

int db_user_set_session(db_t* pDb, const char* username, const char* sesskey_hash, const char* sesskey_salt);

int db_user_unset_session(db_t* pDb, const char* username);

int db_reagent_type_insert(db_t* pDb, const char* name);

int db_reagent_type_insert_ret(db_t* pDb, const char* name, db_reagent_type_t* pReagentType_out);

int db_reagent_type_get_by_id(db_t* pDb, const char* reagtype_id_in, db_reagent_type_t* pReagentType_out);

int db_reagent_insert(db_t* pDb, const char* name, const char* vendor, const char* reagent_type_id);

int db_reagent_insert_ret(db_t* pDb, const char* name, const char* vendor, const char* reagent_type_id, db_reagent_t* pReagent_out);

int db_reagent_get_by_id(db_t* pDb, const char* reagent_id_in, db_reagent_t* pReagent_out);

int db_faculty_insert(db_t* pDb, const char* name, const char* email_domain);

int db_faculty_insert_ret(db_t* pDb, const char* name, const char* email_domain, db_faculty_t* pFaculty_out);

int db_faculty_get_by_id(db_t* pDb, const char* faculty_id_in, db_faculty_t* pFaculty_out);

int db_lab_insert(db_t* pDb, 
                  const char* name, 
                  const char* bearer_token_hash, 
                  const char* bearer_token_salt, 
                  const char* faculty_id);

int db_lab_insert_ret(db_t* pDb, 
                      const char* name, 
                      const char* bearer_token_hash, 
                      const char* bearer_token_salt, 
                      const char* faculty_id, 
                      db_lab_t* pLab_out);

int db_lab_get_by_id(db_t* pDb, const char* lab_id_in, db_lab_t* pLab_out);

int db_inventory_insert(db_t* pDb, 
                        const char* reagent_id, 
                        const char* date_added, 
                        const char* date_expire, 
                        const char* lab_id, 
                        const char* epc);

int db_inventory_insert_ret(db_t* pDb, 
                                 const char* reagent_id, 
                                 const char* date_added, 
                                 const char* date_expire, 
                                 const char* lab_id, 
                                 const char* epc, 
                                 db_inventory_item_t* pInventoryItem_out);                        

int db_inventory_get_by_id(db_t* pDb, const char* inventory_id_in, db_inventory_item_t* pInventoryItem_out);

int db_antenna_insert(db_t* pDb, 
                      const char* name, 
                      const char* info, 
                      const char* k, 
                      const char* lab_id);

int db_antenna_get_by_id(db_t* pDb, const char* antenna_id_in, db_antenna_t* pAntenna_out);

int db_invm_insert(db_t* pDb, 
                   const char* time, 
                   const char* inventory_epc, 
                   const char* antenna_id, 
                   const char* rx_signal_strength, 
                   const char* read_rate, 
                   const char* tx_power, 
                   const char* read_latency, 
                   const char* measurement_type, 
                   const char* rotator_ktheta, 
                   const char* rotator_kphi);

#endif // DB_H