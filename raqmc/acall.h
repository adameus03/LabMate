#ifndef ACALL_H
#define ACALL_H

int acall_ant_set_enabled(const char* path);

int acall_ant_set_disabled(const char* path);

/**
 * @warning Output buffer needs to be freed with free()
 */
const char* acall_ant_get_path(const int index);

#endif // ACALL_H