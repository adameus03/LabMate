#include <hiredis/hiredis.h>
#include "config.h"

#if not defined(RR_REDIS_IP)
#error "RR_REDIS_IP not defined"
#endif
#if not defined(RR_REDIS_PORT)
#error "RR_REDIS_PORT not defined"
#endif


int main() {
  redisReply* pRedisReply = NULL;
  redisContext* pRedisContext = NULL;
  pRedisContext = redisConnect(RR_REDIS_IP, RR_REDIS_PORT);
}// send mails from queue