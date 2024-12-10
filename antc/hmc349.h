#ifndef HMC349_H
#define HMC349_H

typedef struct hmc349_dev hmc349_dev_t; // opaque type for hmc349 device

typedef enum hmc349_outp {
  HMC349_OUTP_RF1,
  HMC349_OUTP_RF2
} hmc349_outp_t;

/**
 * @brief Set output port for hmc349
 */
void hmc349_set_outp(hmc349_dev_t* pDev, hmc349_outp_t outp);

/**
 * @brief Get output port for hmc349
 */
void hmc349_get_outp(hmc349_dev_t* pDev, hmc349_outp_t* pOutp_out);

/**
 * @brief Initialize hmc349 device using underlying libgpiod library
 */
void hmc349_dev_init(hmc349_dev_t* pDev);

/**
 * @brief Deinitialize hmc349 device using underlying libgpiod library
 */
void hmc349_dev_deinit(hmc349_dev_t* pDev);

/**
 * @brief Allocate resources for hmc349 device control structure
 */
hmc349_dev_t* hmc349_dev_new();

/**
 * @brief Free resources allocated by `hmc349_dev_new`
 */
void hmc349_dev_free(hmc349_dev_t* pDev);

#endif // HMC349_H