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
 * @note `pWalker` is expected to be NULL. It can be later used to stop the walker thread. 
 * @warning Does not free `pWalker` resources allocated by `walker_start_thread`. You should call `walker_free_resources` to free them.
 * @warning `walker_init_antenna_table` should be called before calling this function
 */
void walker_start_thread(walker_t* pWalker);

/**
 * @warning Does not free `pWalker` resources allocated by `walker_start_thread`. You should call `walker_free_resources` to free them.
 */
void walker_stop_thread(walker_t* pWalker);

/**
 * @brief Free `pWalker` resources allocated by `walker_start_thread` 
 */
void walker_free_resources(walker_t* pWalker);

#endif // WALKER_H