#include "hmc349.h"
#include <gpiod.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>
#include "log.h"
#include "config.h"

#define HMC349_GPIOD_LINE_CONSUMER_NAME "hmc349"

typedef enum hmc349_vctl_val {
  LOW = 0,
  HIGH = 1
} hmc349_vctl_val_t;

struct hmc349_dev {
  hmc349_vctl_val_t vctl;
  struct {
    struct gpiod_chip* chip;
    struct gpiod_line* line;
  } w;
};

void hmc349_set_outp(hmc349_dev_t* pDev, hmc349_outp_t outp) {
  if (!pDev) {
    LOG_E("hmc349_set_outp: pDev is NULL");
    exit(EXIT_FAILURE);
  }
  int rv = gpiod_line_set_value(pDev->w.line, outp);
  if (0 != rv) {
    LOG_E("hmc349_set_outp: Failed to set value %d for line %d for GPIO chip %s with consumer name %s (rv: %d, errno: %d)", outp, HMC349_GPIOD_CHIP_LINE, HMC349_GPIOD_CHIP, HMC349_GPIOD_LINE_CONSUMER_NAME, rv, errno);
    exit(EXIT_FAILURE);
  }
}

void hmc349_get_outp(hmc349_dev_t* pDev, hmc349_outp_t* pOutp_out) {
  if (!pDev) {
    LOG_E("hmc349_get_outp: pDev is NULL");
    exit(EXIT_FAILURE);
  }
  if (pOutp_out) {
    int rv = gpiod_line_get_value(pDev->w.line);
    if (rv < 0) {
      LOG_E("hmc349_get_outp: Failed to get value for line %d for GPIO chip %s with consumer name %s and default value %d (rv: %d, errno: %d)", HMC349_GPIOD_CHIP_LINE, HMC349_GPIOD_CHIP, HMC349_GPIOD_LINE_CONSUMER_NAME, LOW, rv, errno);
      exit(EXIT_FAILURE);
    }
    assert((rv == LOW) || (rv == HIGH));
    *pOutp_out = (hmc349_outp_t)rv;
  } else {
    LOG_W("hmc349_get_outp: pOutp_out is NULL");
  }
}

void hmc349_dev_init(hmc349_dev_t* pDev) {
  if (!pDev) {
    LOG_F("hmc349_dev_init: pDev is NULL");
    exit(EXIT_FAILURE);
  }
  pDev->w.chip = gpiod_chip_open(HMC349_GPIOD_CHIP);
  if (!pDev->w.chip) {
    LOG_F("hmc349_dev_init: Failed to open GPIO chip [%s] (errno: %d)", HMC349_GPIOD_CHIP, errno);
    exit(EXIT_FAILURE);
  }
  pDev->w.line = gpiod_chip_get_line(pDev->w.chip, HMC349_GPIOD_CHIP_LINE);
  if (!pDev->w.line) {
    LOG_F("hmc349_dev_init: Failed to get line %d for GPIO chip %s (errno: %d)", HMC349_GPIOD_CHIP_LINE, HMC349_GPIOD_CHIP, errno);
    exit(EXIT_FAILURE);
  }
  int rv = gpiod_line_request_output(pDev->w.line, HMC349_GPIOD_LINE_CONSUMER_NAME, LOW);
  if (0 != rv) {
    LOG_F("hmc349_dev_init: Failed to request output for line %d for GPIO chip %s with consumer name %s and default value %d (rv: %d, errno: %d)", HMC349_GPIOD_CHIP_LINE, HMC349_GPIOD_CHIP, HMC349_GPIOD_LINE_CONSUMER_NAME, LOW, rv, errno);
    exit(EXIT_FAILURE);
  }
  rv = gpiod_line_get_value(pDev->w.line);
  if (rv < 0) {
    LOG_F("hmc349_dev_init: Failed to get value for line %d for GPIO chip %s with consumer name %s and default value %d (rv: %d, errno: %d)", HMC349_GPIOD_CHIP_LINE, HMC349_GPIOD_CHIP, HMC349_GPIOD_LINE_CONSUMER_NAME, LOW, rv, errno);
    exit(EXIT_FAILURE);
  }
  assert((rv == LOW) || (rv == HIGH));
  pDev->vctl = (hmc349_vctl_val_t)rv;
  LOG_D("hmc349_dev_init: Initialized hmc349 device with GPIO chip %s, line %d, consumer name %s, default value %d. Current value: %d", HMC349_GPIOD_CHIP, HMC349_GPIOD_CHIP_LINE, HMC349_GPIOD_LINE_CONSUMER_NAME, LOW, pDev->vctl);

}

void hmc349_dev_deinit(hmc349_dev_t* pDev) {
  if (!pDev) {
    LOG_W("hmc349_dev_deinit: pDev is NULL");
    return;
  }
  if (pDev->w.line) {
    gpiod_line_release(pDev->w.line);
  } else {
    LOG_W("hmc349_dev_deinit: pDev->w.line is NULL");
  }

  if (pDev->w.chip) {
    gpiod_chip_close(pDev->w.chip);
  } else {
    LOG_W("hmc349_dev_deinit: pDev->w.chip is NULL");
  }
  
  LOG_D("hmc349_dev_deinit: Deinitialized hmc349 device with GPIO chip %s, line %d, consumer name %s, default value %d", HMC349_GPIOD_CHIP, HMC349_GPIOD_CHIP_LINE, HMC349_GPIOD_LINE_CONSUMER_NAME, LOW);
}