struct hmc349_dev_t;

typedef enum {
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