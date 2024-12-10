#include "hmc349.h"

typedef enum {
  LOW = 0,
  HIGH = 1
} hmc349_vctl_val_t;

typedef struct {
  hmc349_vctl_val_t vctl;
} hmc349_dev_t;

void hmc349_set_outp(hmc349_dev_t* pDev, hmc349_outp_t outp) {

}

void hmc349_get_outp(hmc349_dev_t* pDev, hmc349_outp_t* pOutp_out) {
  
}