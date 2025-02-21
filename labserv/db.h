#ifndef DB_H
#define DB_H

#include <stdlib.h>

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
  char* lab_key;
  char* host;
  int faculty_id;
} db_lab_t;

typedef struct db_reagent {
  int reagent_id;
  char* name;
  int reagent_type_id;
  char* vendor;
} db_reagent_t;

typedef struct db_vendor {
  int vendor_id;
  char* name;
} db_vendor_t;

typedef struct db_inventory_item {
  int inventory_id;
  int reagent_id;
  char* date_added;
  char* date_expire;
  int lab_id;
  char* epc;
  char* apwd;
  char* kpwd;
  int is_embodied;
  int basepoint_id;
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
  int antno;
  int rx_signal_strength;
  int read_rate;
  int tx_power;
  int read_latency;
  int measurement_type;
  int rotator_ktheta; // For future when antennas can rotate
  int rotator_kphi; // For future when antennas can rotate
} db_invm_t;

void db_user_free(db_user_t* pUser);
void db_faculty_free(db_faculty_t* pFaculty);
void db_reagent_type_free(db_reagent_type_t* pReagentType);
void db_lab_free(db_lab_t* pLab);
void db_reagent_free(db_reagent_t* pReagent);
void db_vendor_free(db_vendor_t* pVendor);
void db_inventory_item_free(db_inventory_item_t* pInventoryItem);
void db_antenna_free(db_antenna_t* pAntenna);
void db_invm_free(db_invm_t* pInvm);

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

int db_reagent_types_get_total_count(db_t* pDb, int* pCount_out);

typedef enum db_reagent_type_filter_type {
  DB_REAGENT_TYPE_FILTER_TYPE_NONE,
  DB_REAGENT_TYPE_FILTER_TYPE_NAME
} db_reagent_type_filter_type_t;

int db_reagent_types_read_page_filtered(db_t* pDb, 
                                        const char* offset, 
                                        const char* page_size, 
                                        db_reagent_type_t** ppReagentTypes_out, 
                                        int* pN_out, 
                                        db_reagent_type_filter_type_t filter_type, 
                                        const char* filter_value);

int db_reagent_insert(db_t* pDb, const char* name, const char* vendor, const char* reagent_type_id);

int db_reagent_insert_ret(db_t* pDb, const char* name, const char* vendor, const char* reagent_type_id, db_reagent_t* pReagent_out);

int db_reagent_get_by_id(db_t* pDb, const char* reagent_id_in, db_reagent_t* pReagent_out);

int db_reagents_get_total_count(db_t* pDb, int* pCount_out);

int db_reagents_read_page(db_t* pDb, const char* offset, const char* page_size, db_reagent_t** ppReagents_out, int* pN_out);

typedef enum db_reagent_filter_type {
  DB_REAGENT_FILTER_TYPE_NONE,
  DB_REAGENT_FILTER_TYPE_NAME,
  DB_REAGENT_FILTER_TYPE_VENDOR,
  DB_REAGENT_FILTER_TYPE_REAGTYPE_ID,
  DB_REAGENT_FILTER_TYPE_REAGTYPE_NAME
} db_reagent_filter_type_t;

/**
 * @brief Read a page of reagents from the database with filtering using regex on specified attribute
 */
int db_reagents_read_page_filtered(db_t* pDb, 
                                   const char* offset, 
                                   const char* page_size, 
                                   db_reagent_t** ppReagents_out, 
                                   int* pN_out, 
                                   db_reagent_filter_type_t filter_type, 
                                   const char* filter_value);

int db_vendor_get_by_id(db_t* pDb, const char* vendor_id_in, db_vendor_t* pVendor_out);

int db_vendors_get_total_count(db_t* pDb, int* pCount_out);

typedef enum db_vendor_filter_type {
  DB_VENDOR_FILTER_TYPE_NONE,
  DB_VENDOR_FILTER_TYPE_NAME
} db_vendor_filter_type_t;

int db_vendors_read_page_filtered(db_t* pDb, 
                                  const char* offset, 
                                  const char* page_size, 
                                  db_vendor_t** ppVendors_out, 
                                  int* pN_out, 
                                  db_vendor_filter_type_t filter_type,
                                  const char* filter_value);


int db_faculty_insert(db_t* pDb, const char* name, const char* email_domain);

int db_faculty_insert_ret(db_t* pDb, const char* name, const char* email_domain, db_faculty_t* pFaculty_out);

int db_faculty_get_by_id(db_t* pDb, const char* faculty_id_in, db_faculty_t* pFaculty_out);

int db_faculties_get_total_count(db_t* pDb, int* pCount_out);

typedef enum db_faculty_filter_type {
  DB_FACULTY_FILTER_TYPE_NONE,
  DB_FACULTY_FILTER_TYPE_NAME,
  DB_FACULTY_FILTER_TYPE_USER_ID,
  DB_FACULTY_FILTER_TYPE_EMAIL_DOMAIN
} db_faculty_filter_type_t;

int db_faculties_read_page_filtered(db_t* pDb, 
                                    const char* offset, 
                                    const char* page_size, 
                                    db_faculty_t** ppFaculties_out, 
                                    int* pN_out, 
                                    db_faculty_filter_type_t filter_type, 
                                    const char* filter_value);

int db_lab_insert(db_t* pDb, 
                  const char* name, 
                  const char* bearer_token_hash, 
                  const char* bearer_token_salt,
                  const char* lab_key,
                  const char* host,
                  const char* faculty_id);

int db_lab_insert_ret(db_t* pDb, 
                      const char* name, 
                      const char* bearer_token_hash, 
                      const char* bearer_token_salt, 
                      const char* lab_key,
                      const char* host,
                      const char* faculty_id, 
                      db_lab_t* pLab_out);

int db_lab_get_by_id(db_t* pDb, const char* lab_id_in, db_lab_t* pLab_out);

int db_lab_get_by_epc(db_t* pDb, const char* epc_in, db_lab_t* pLab_out);

int db_lab_get_by_host(db_t* pDb, const char* host_in, db_lab_t* pLab_out);

int db_labs_get_total_count(db_t* pDb, int* pCount_out);

typedef enum db_lab_filter_type {
  DB_LAB_FILTER_TYPE_NONE,
  DB_LAB_FILTER_TYPE_NAME,
  DB_LAB_FILTER_TYPE_FACULTY_ID,
  DB_LAB_FILTER_TYPE_USER_ID
} db_lab_filter_type_t;

int db_labs_read_page_filtered(db_t* pDb, 
                               const char* offset, 
                               const char* page_size, 
                               db_lab_t** ppLabs_out, 
                               int* pN_out, 
                               db_lab_filter_type_t filter_type, 
                               const char* filter_value);

int db_inventory_insert(db_t* pDb, 
                        const char* reagent_id, 
                        const char* date_added, 
                        const char* date_expire, 
                        const char* lab_id, 
                        const char* epc,
                        const char* apwd,
                        const char* kpwd);

int db_inventory_insert_ret(db_t* pDb, 
                                 const char* reagent_id, 
                                 const char* date_added, 
                                 const char* date_expire, 
                                 const char* lab_id, 
                                 const char* epc, 
                                 const char* apwd,
                                 const char* kpwd,
                                 db_inventory_item_t* pInventoryItem_out);                        

int db_inventory_get_by_id(db_t* pDb, const char* inventory_id_in, db_inventory_item_t* pInventoryItem_out);

//int db_inventory_get_by_lab_bthash(db_t* pDb, const char* lab_bthash_in, db_inventory_item_t* pInventoryItems_out, size_t* pN_out);

/**
 * @note This function allocates memory for `*ppInventoryItems_out` which must be freed by the caller after use
 */
int db_inventory_get_by_lab_host(db_t* pDb, const char* lab_host_in, db_inventory_item_t** ppInventoryItems_out, size_t* pN_out);

/**
 * @note This function allocates memory for `*ppInventoryItems_out` which must be freed by the caller after use
 */
int db_inventory_get_by_lab_id(db_t* pDb, const char* lab_id_in, db_inventory_item_t** ppInventoryItems_out, size_t* pN_out);

int db_inventory_get_by_lab_id_filter_embodied(db_t* pDb, const char* lab_id_in, const int is_embodied_in, db_inventory_item_t** ppInventoryItems_out, size_t* pN_out);

int db_inventory_set_embodied(db_t* pDb, const char* inventory_id);

int db_inventory_items_get_total_count(db_t* pDb, int* pCount_out);

typedef enum db_inventory_item_filter_type {
  DB_INVENTORY_ITEM_FILTER_TYPE_NONE,
  DB_INVENTORY_ITEM_FILTER_TYPE_REAGENT_ID,
  DB_INVENTORY_ITEM_FILTER_TYPE_LAB_ID,
  DB_INVENTORY_ITEM_FILTER_TYPE_EPC,
  DB_INVENTORY_ITEM_FILTER_TYPE_FACULTY_ID,
  DB_INVENTORY_ITEM_FILTER_TYPE_RID_LID,
  DB_INVENTORY_ITEM_FILTER_TYPE_RID_FID,
  DB_INVENTORY_ITEM_FILTER_TYPE_IS_EMBODIED,
} db_inventory_item_filter_type_t;


int db_inventory_items_read_page_filtered(db_t* pDb, 
                                          const char* offset, 
                                          const char* page_size, 
                                          db_inventory_item_t** ppInventoryItems_out, 
                                          int* pN_out, 
                                          db_inventory_item_filter_type_t filter_type, 
                                          const char* filter_value);

int db_antenna_insert(db_t* pDb, 
                      const char* name, 
                      const char* info, 
                      const char* k, 
                      const char* lab_id);

int db_antenna_insert_ret(db_t* pDb, 
                          const char* name, 
                          const char* info, 
                          const char* k, 
                          const char* lab_id, 
                          db_antenna_t* pAntenna_out);

int db_antenna_get_by_id(db_t* pDb, const char* antenna_id_in, db_antenna_t* pAntenna_out);

int db_invm_insert(db_t* pDb, 
                   const char* time, 
                   const char* inventory_epc, 
                   const char* antno, 
                   const char* rx_signal_strength, 
                   const char* read_rate, 
                   const char* tx_power, 
                   const char* read_latency, 
                   const char* measurement_type, 
                   const char* rotator_ktheta, 
                   const char* rotator_kphi);

int db_invm_insert_bulk(db_t* pDb, 
                        const size_t nInvms, 
                        const char* times[],
                        const char* inventory_epcs[],
                        const char* antnos[],
                        const char* rx_signal_strengths[],
                        const char* read_rates[],
                        const char* tx_powers[],
                        const char* read_latencies[],
                        const char* measurement_types[],
                        const char* rotator_kthetas[],
                        const char* rotator_kphis[]);

int db_invm_insert_ret(db_t* pDb, 
                       const char* time, 
                       const char* inventory_epc, 
                       const char* antno,
                       const char* rx_signal_strength, 
                       const char* read_rate, 
                       const char* tx_power, 
                       const char* read_latency, 
                       const char* measurement_type, 
                       const char* rotator_ktheta, 
                       const char* rotator_kphi, 
                       db_invm_t* pInvm_out);

// int db_invm_insert_bulk_ret(db_t* pDb, 
//                         const size_t nInvms, 
//                         const char* times[],
//                         const char* inventory_epcs[],
//                         const char* antnos[],
//                         const char* rx_signal_strengths[],
//                         const char* read_rates[],
//                         const char* tx_powers[],
//                         const char* read_latencies[],
//                         const char* measurement_types[],
//                         const char* rotator_kthetas[],
//                         const char* rotator_kphis[],
//                         db_invm_t** ppInvms_out);           

#endif // DB_H