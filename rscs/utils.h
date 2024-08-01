#include <stdint.h>
#include <stddef.h>

/*
    CRC-16-CCITT from the Linux kernel
    @source https://elixir.bootlin.com/linux/v6.10.2/source/include/linux/crc-ccitt.h
    @source https://elixir.bootlin.com/linux/v6.10.2/source/lib/crc-ccitt.c
*/

/**
 *	crc_ccitt - recompute the CRC (CRC-CCITT variant) for the data
 *	buffer
 *	@crc: previous CRC value
 *	@buffer: data pointer
 *	@len: number of bytes in the buffer
 */
uint16_t utils_crc_ccitt(uint16_t crc, uint8_t const *buffer, size_t len);

//uint16_t utils_crc_ccitt_genibus(uint8_t const *buffer, size_t len);

void utils_buf_u8_to_u16_big_endian(uint16_t *dst, const uint8_t *src, size_t u8len);