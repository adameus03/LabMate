#ifndef WALKER_H
#define WALKER_H

typedef struct walker walker_t;

/**
 * @brief Fetch the antenna table from the labserv host endpoint
 * @note OBSOLETE (now antno is sent instead of ant_id during measurement transmission)
 */
//void walker_init_antenna_table(walker_t* pWalker);

/**
 * @brief Start a new walker instance in a separate thread
 * @warning You should free resources by calling `walker_free_resources` on the returned pointer. It is recommended to call `walker_stop_thread` before freeing resources.
 */
walker_t* walker_start_thread(void);

/**
 * @warning Does not free `pWalker` resources allocated by `walker_start_thread`. You should call `walker_free_resources` to free them.
 */
void walker_stop_thread(walker_t* pWalker);

/**
 * @brief Free `pWalker` resources allocated by `walker_start_thread` 
 */
void walker_free_resources(walker_t* pWalker);

#endif // WALKER_H