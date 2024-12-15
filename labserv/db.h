#ifndef DB_H
#define DB_H

typedef struct db db_t;

/**
 * @brief Initialize database driver
 */
void db_init(db_t* pDb);

#endif // DB_H