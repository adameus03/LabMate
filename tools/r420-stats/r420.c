#include <sys/socket.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include "r420.h"

#if defined(__linux__)
#  include <endian.h>
#elif defined(__FreeBSD__) || defined(__NetBSD__)
#  include <sys/endian.h>
#elif defined(__OpenBSD__)
#  include <sys/types.h>
#  define be16toh(x) betoh16(x)
#  define be32toh(x) betoh32(x)
#  define be64toh(x) betoh64(x)
#endif
static uint64_t ntohll(uint64_t val) {
  return be64toh(val);
}

#define R420_MSG_BODY_MAX_SIZE 4096
#define IMPINJ_VENDOR_ID R420_CUSTOM_MESSAGE_VENDOR_ID_IMPINJ
#define REPLACE_ROSPEC 1

typedef struct r420_msg_body {
  uint8_t buf[R420_MSG_BODY_MAX_SIZE];
  size_t len;
} r420_msg_body_t;

typedef struct r420_msg_body_param_tlv_hdr {
  uint16_t attrs;
  uint16_t param_len;
} __attribute__((__packed__)) r420_msg_body_param_tlv_hdr_t;

// reserved: bits 15-10
#define R420_MSG_BODY_PARAM_TLV_HDR_RESERVED(hdr) ((ntohs((hdr).attrs) >> 10) & 0x3F)
// type: bits 9-0
#define R420_MSG_BODY_PARAM_TLV_HDR_TYPE(hdr) (ntohs((hdr).attrs) & 0x3FF)
#define R420_MSG_BODY_PARAM_TLV_HDR_LEN(hdr) (ntohs((hdr).param_len))

typedef struct r420_msg_body_param_tv_hdr {
  uint8_t attrs;
} __attribute__((__packed__)) r420_msg_body_param_tv_hdr_t;

// reserved: bit 7
#define R420_MSG_BODY_PARAM_TV_HDR_RESERVED(hdr) ((ntohs((hdr).attrs) >> 7) & 1)
// type: bits 6-0
#define R420_MSG_BODY_PARAM_TV_HDR_TYPE(hdr) ((hdr).attrs & 0x7F)

typedef struct r420_msg_body_param_utc_timestamp_value {
  uint64_t microseconds;
} __attribute__((__packed__)) r420_msg_body_param_utc_timestamp_value_t;

void r420_log(const r420_ctx_t* pCtx, const char* pMsg) {
  if (pCtx->log_handler) {
    pCtx->log_handler(pCtx, pMsg);
  }
}

void r420_logf(const r420_ctx_t* pCtx, const char* fmt, ...) {
  if (pCtx->log_handler) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    pCtx->log_handler(pCtx, buf);
  }
}

r420_ctx_t r420_connect(const r420_connection_parameters_t conn_params) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  assert(fd >= 0);
  struct sockaddr_in addr = {
    .sin_family = AF_INET,
    .sin_port = htons(conn_params.port),  // Convert to network byte order
    .sin_addr.s_addr = htonl(conn_params.ip)  // Convert to network byte order
  };
  assert(0 == connect(fd, (struct sockaddr *)&addr, sizeof(addr)));
  return (r420_ctx_t){ .fd = fd, .loop_handler = NULL, .log_handler = NULL, .next_tx_msg_id = 1, .llrp_version = 1, .terminate_flag = 0, .rospec_added = 0,  .rospec_replace_flag = 0, .rospec_enabled = 0, .rospec_started = 0  };
}

void r420_ctx_set_loop_handler(r420_ctx_t *pCtx, r420_loop_callback_t loop_handler) {
  pCtx->loop_handler = loop_handler;
}

void r420_ctx_set_log_handler(r420_ctx_t *pCtx, r420_log_callback_t log_handler) {
  pCtx->log_handler = log_handler;
}

void r420_close(r420_ctx_t *pCtx) {
  assert(0 == close(pCtx->fd));
  pCtx->fd = -1;
}

r420_msg_hdr_t r420_receive_header(const r420_ctx_t *pCtx) {
  r420_msg_hdr_t hdr = {0};
  ssize_t remaining_rcv_bytes = sizeof(hdr);
  while (remaining_rcv_bytes > 0) {
    ssize_t n = read(pCtx->fd, ((uint8_t *)&hdr) + (sizeof(hdr) - remaining_rcv_bytes), remaining_rcv_bytes);
    assert(n > 0);
    remaining_rcv_bytes -= n;
  }
  return hdr;
}

r420_msg_body_t r420_receive_body(const r420_ctx_t *pCtx, size_t body_len) {
  assert(body_len <= R420_MSG_BODY_MAX_SIZE);
  r420_msg_body_t body = { .len = body_len, .buf = {0} };
  ssize_t remaining_rcv_bytes = body_len;
  while (remaining_rcv_bytes > 0) {
    ssize_t n = read(pCtx->fd, body.buf + (body_len - remaining_rcv_bytes), remaining_rcv_bytes);
    assert(n > 0);
    remaining_rcv_bytes -= n;
  }
  return body;
}

typedef struct r420_msg_param_info {
  uint16_t type; // parameter type as per LLRP specification
  uint16_t len; // total length of the parameter including header
  uint16_t hdr_len; // Length of the parameter header. TODO This is redundant as hdr_len=value_offset - hdr_offset
  size_t hdr_offset; // Offset of the parameter header within the message body
  size_t value_offset; // Offset of the value field within the ~~parameter~~ message body (after header)
} r420_msg_param_info_t;

r420_msg_param_info_t r420_process_param(const r420_ctx_t* pCtx, const r420_msg_body_t *pBody, size_t offset) {
  assert(offset < pBody->len);
  if (pBody->buf[offset] & 0x80) { // check if the parameter is TLV or TV encoded
    // TV encoded
    r420_msg_body_param_tv_hdr_t tv_hdr = *(r420_msg_body_param_tv_hdr_t *)(pBody->buf + offset);
    uint8_t type = R420_MSG_BODY_PARAM_TV_HDR_TYPE(tv_hdr);
    uint16_t len = 0xffff; // TV has no length field
    switch (type) {
      // TODO We should handle all TV parameter lengths here
      case R420_PARAM_TYPE_PEAK_RSSI:
        return (r420_msg_param_info_t){ .type = type, .len = 2, .hdr_len = sizeof(tv_hdr), .hdr_offset = offset, .value_offset = offset + sizeof(tv_hdr) };
      case R420_PARAM_TYPE_EPC_96:
        return (r420_msg_param_info_t){ .type = type, .len = 13, .hdr_len = sizeof(tv_hdr), .hdr_offset = offset, .value_offset = offset + sizeof(tv_hdr) };
      case R420_PARAM_TYPE_ANTENNA_ID:
        return (r420_msg_param_info_t){ .type = type, .len = 3, .hdr_len = sizeof(tv_hdr), .hdr_offset = offset, .value_offset = offset + sizeof(tv_hdr) };
      case R420_PARAM_TYPE_FIRST_SEEN_TIMESTAMP_UTC:
        return (r420_msg_param_info_t){ .type = type, .len = 9, .hdr_len = sizeof(tv_hdr), .hdr_offset = offset, .value_offset = offset + sizeof(tv_hdr) };
      case R420_PARAM_TYPE_LAST_SEEN_TIMESTAMP_UTC:
        return (r420_msg_param_info_t){ .type = type, .len = 9, .hdr_len = sizeof(tv_hdr), .hdr_offset = offset, .value_offset = offset + sizeof(tv_hdr) };
      case R420_PARAM_TYPE_TAG_SEEN_COUNT:
        return (r420_msg_param_info_t){ .type = type, .len = 3, .hdr_len = sizeof(tv_hdr), .hdr_offset = offset, .value_offset = offset + sizeof(tv_hdr) };
        case R420_PARAM_TYPE_CHANNEL_INDEX:
        return (r420_msg_param_info_t){ .type = type, .len = 3, .hdr_len = sizeof(tv_hdr), .hdr_offset = offset, .value_offset = offset + sizeof(tv_hdr) };
      default:
        r420_logf(pCtx, "r420_process_param: Unknown TV parameter type 0x%X, cannot determine length", type);
        break;
    }
    return (r420_msg_param_info_t){ .type = type, .len = 0xffff, .hdr_len = 0xffff, .hdr_offset = offset, .value_offset = offset + sizeof(tv_hdr) }; // TV has no length field
  } else {
    // TLV encoded
    r420_msg_body_param_tlv_hdr_t tlv_hdr = *(r420_msg_body_param_tlv_hdr_t *)(pBody->buf + offset);
    uint16_t type = R420_MSG_BODY_PARAM_TLV_HDR_TYPE(tlv_hdr);
    uint16_t len = R420_MSG_BODY_PARAM_TLV_HDR_LEN(tlv_hdr);
    return (r420_msg_param_info_t){ .type = type, .len = len, .hdr_len = sizeof(tlv_hdr), .hdr_offset = offset, .value_offset = offset + sizeof(tlv_hdr) };
  }
}

// void r420_parse_reader_event_notification_data_param_value(const r420_msg_body_t* pBody, size_t offset) {

// }



void r420_process_reader_event_notification_msg(const r420_ctx_t *pCtx, const r420_msg_body_t *pBody) {
  // First 4 bytes - TLV header for ReaderEventNotificationData parameter
  // Next 12 bytes - TimestampParameter (either UTCTimestamp or Uptime)
  // Next params are optional, so pBody->len should be at least 16 bytes long

  assert(pBody->len >= 16);
  //We are lazy and we don't do anything with most of the parameters for now
  //Usually, at start of communication, a UTCTimestamp parameter and a ConnectionAttemptEvent parameter with status=Success are present

  //Handle selected parameters in a loop
  size_t offset = 16; // Start after the mandatory parameters
  while (offset < pBody->len) {
    r420_msg_param_info_t param_info = r420_process_param(pCtx, pBody, offset);
    switch(param_info.type) {
      case R420_PARAM_TYPE_UTC_TIMESTAMP:
        break;
      case R420_PARAM_TYPE_CONNECTION_ATTEMPT_EVENT:
        uint16_t status = *(uint16_t *)(pBody->buf + param_info.value_offset);
        assert(status == 0); // We expect status=Success
        break;
      default:
        // Unknown parameter type, we just skip it
        r420_logf(pCtx, "r420_process_reader_event_notification_msg: Skipping unknown parameter type 0x%X", param_info.type);
        //TODO Add a ctx handler to pass this event to the user
        break;
    }
    assert(param_info.len != 0xffff); // Make sure there are no unsupported TV params as that would break the parsing TODO
    offset += param_info.len;
  }
}

void r420_process_get_supported_version_response_msg(const r420_ctx_t *pCtx, const r420_msg_body_t *pBody) {
  // First byte - CurrentVersion
  // Next byte - SupportedVersion
  // Next 4 bytes - TLV header for LLRPStatus parameter
  // Next 2 bytes - StatusCode
  // Next 2 bytes - Error Description ByteCount
  // ...
  assert(pBody->len >= 10); // We expect at least 10 bytes
  uint8_t current_version = pBody->buf[0];
  uint8_t supported_version = pBody->buf[1];
  r420_logf(pCtx, "R420 Protocol Version: Current=%u, Supported=%u", current_version, supported_version);
  assert(current_version == 2); // We expect version 2
  assert(supported_version == 2); // We expect version 2
  r420_msg_param_info_t param_info = r420_process_param(pCtx, pBody, 2);
  assert(param_info.type == R420_PARAM_TYPE_LLRP_STATUS);
  assert(param_info.len >= 4); // We expect at least 4 bytes for LLRPStatus
  uint16_t status_code = *(uint16_t *)(pBody->buf + param_info.value_offset);
  assert(status_code == 0); // We expect StatusCode=M_Success
  //uint16_t err_desc_len = *(uint16_t *)(pBody->buf + param_info.value_offset + 2);
  // We ignore the error description and error params for now
  // TODO Handle the error description and error params
}

void r420_process_get_reader_capabilities_response_msg(const r420_ctx_t *pCtx, const r420_msg_body_t *pBody) {
  // First 4 bytes - TLV header for LLRPStatus parameter
  // Next 2 bytes - StatusCode
  // Next 2 bytes - Error Description ByteCount
  // ...
  assert(pBody->len >= 8); // We expect at least 8 bytes

  size_t offset = 0;
  while (offset < pBody->len) {
    r420_msg_param_info_t param_info = r420_process_param(pCtx, pBody, offset);
    switch(param_info.type) {
      case R420_PARAM_TYPE_LLRP_STATUS: {
        r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: LLRPStatus begins at offset %zu", offset);
        assert(param_info.len >= 4); // We expect at least 4 bytes for LLRPStatus
        uint16_t status_code = *(uint16_t *)(pBody->buf + param_info.value_offset);
        assert(status_code == 0); // We expect StatusCode=M_Success
        //uint16_t err_desc_len = *(uint16_t *)(pBody->buf + param_info.value_offset + 2);
        // We ignore the error description and error params for now
        // TODO Handle the error description and error params
        break;
      }
      case R420_PARAM_TYPE_GENERAL_DEVICE_CAPABILITIES: {
        r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: GeneralDeviceCapabilities begins at offset %zu", offset);
        // We can parse specific fields if needed
        break;
      }
      case R420_PARAM_TYPE_LLRP_CAPABILITIES: {
        r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: LLRPCapabilities begins at offset %zu", offset);
        // We can parse specific fields if needed
        break;
      }
      case R420_PARAM_TYPE_REGULATORY_CAPABILITIES: {
        r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: RegulatoryCapabilities begins at offset %zu", offset);
        assert(param_info.len >= 4); // We expect at least 4 bytes for the RegulatoryCapabilities parameter
        size_t suboffset = 0;
        uint16_t country_code = ntohs(*(uint16_t *)(pBody->buf + param_info.value_offset + suboffset)); // See ISO 3166, 0 means unspecified
        suboffset += sizeof(country_code);
        uint16_t communications_standard = ntohs(*(uint16_t *)(pBody->buf + param_info.value_offset + suboffset));
        suboffset += sizeof(communications_standard);
        // Process subparameters
        while (param_info.value_offset + suboffset < param_info.hdr_offset + param_info.len) {
          r420_msg_param_info_t subparam_info = r420_process_param(pCtx, pBody, param_info.value_offset + suboffset);
          switch(subparam_info.type) {
            case R420_PARAM_TYPE_UHF_BAND_CAPABILITIES:
              r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: UHFBandCapabilities begins at suboffset %zu\n", suboffset);
              assert(subparam_info.len >= 0);
              size_t subsuboffset = 0;
              while (subparam_info.value_offset + subsuboffset < subparam_info.hdr_offset + subparam_info.len) {
                r420_msg_param_info_t subsubparam_info = r420_process_param(pCtx, pBody, subparam_info.value_offset + subsuboffset);
                switch(subsubparam_info.type) {
                  case R420_PARAM_TYPE_TRANSMIT_POWER_LEVL_TABLE_ENTRY:
                    r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: TransmitPowerLevelTableEntry begins at subsuboffset %zu", subsuboffset);
                    // We can parse specific fields if needed
                    break;
                  case R420_PARAM_TYPE_FREQUENCY_INFORMATION:
                    r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: FrequencyInformation begins at subsuboffset %zu", subsuboffset);
                    size_t subsubsuboffset = 0;
                    uint8_t flags = *(uint8_t *)(pBody->buf + subsubparam_info.value_offset);
                    subsubsuboffset += sizeof(flags);
                    if (flags & 0x80) {
                      r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: Hopping flag is SET in FrequencyInformation");
                    } else {
                      r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: Hopping flag is NOT set in FrequencyInformation");
                    }
                    while (subsubparam_info.value_offset + subsubsuboffset < subsubparam_info.hdr_offset + subsubparam_info.len) {
                      r420_msg_param_info_t subsubsubparam_info = r420_process_param(pCtx, pBody, subsubparam_info.value_offset + subsubsuboffset);
                      switch(subsubsubparam_info.type) {
                        case R420_PARAM_TYPE_FREQUENCY_HOP_TABLE:
                          r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: FrequencyHopTable begins at subsubsuboffset %zu\n", subsubsuboffset);
                          size_t subsubsubsuboffset = 0;
                          uint8_t hop_table_id = *(uint8_t *)(pBody->buf + subsubsubparam_info.value_offset);
                          subsubsubsuboffset += sizeof(hop_table_id);
                          uint8_t reserved = *(uint8_t *)(pBody->buf + subsubsubparam_info.value_offset + subsubsubsuboffset);
                          subsubsubsuboffset += sizeof(reserved);
                          uint16_t num_hops = ntohs(*(uint16_t *)(pBody->buf + subsubsubparam_info.value_offset + subsubsubsuboffset));
                          subsubsubsuboffset += sizeof(num_hops);
                          r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: FrequencyHopTable ID=%u, NumHops=%u", hop_table_id, num_hops);
                          r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: Current position in body: %zu, end of FrequencyHopTable at %zu", subsubsubparam_info.value_offset + subsubsubsuboffset, subsubsubparam_info.hdr_offset + subsubsubparam_info.len);
                          assert(subsubsubparam_info.value_offset + subsubsubsuboffset + num_hops * sizeof(uint32_t) == subsubsubparam_info.hdr_offset + subsubsubparam_info.len);
                          for (int i = 0; i < num_hops; i++) {
                            uint32_t frequency = ntohl(*(uint32_t *)(pBody->buf + subsubsubparam_info.value_offset + subsubsubsuboffset + i * sizeof(uint32_t)));
                            r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: FrequencyHopTable Entry %d: Frequency = %u kHz", i, frequency);
                          }
                          break;
                        case R420_PARAM_TYPE_FIXED_FREQUENCY_TABLE:
                          r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: Found FixedFrequencyTable in FrequencyInformation, which is unsupported by this implementation");
                          // We can parse specific fields if needed
                          break;
                        default:
                          // Unknown sub-sub-sub-parameter type, we just skip it
                          r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: Skipping unknown sub-sub-sub-parameter type 0x%X in FrequencyInformation", subsubsubparam_info.type);
                          //TODO Add a ctx handler to pass this event to the user
                          break;
                      }
                      assert(subsubsubparam_info.len != 0xffff); // Make sure there are no unsupported TV params as that would break the parsing TODO
                      subsubsuboffset += subsubsubparam_info.len;
                    }
                    
                    break;
                  case R420_PARAM_TYPE_UHF_C1G2_RF_MODE_TABLE:
                    r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: UHF_C1G2_RFModeTable begins at subsuboffset %zu", subsuboffset);
                    // We can parse specific fields if needed
                    break;
                  default:
                    // Unknown sub-sub-parameter type, we just skip it
                    r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: Skipping unknown sub-sub-parameter type 0x%X in UHFBandCapabilities", subsubparam_info.type);
                    //TODO Add a ctx handler to pass this event to the user
                    break;
                }
                assert(subsubparam_info.len != 0xffff); // Make sure there are no unsupported TV params as that would break the parsing TODO
                subsuboffset += subsubparam_info.len;
              }
              break;
            default:
              // Unknown sub-parameter type, we just skip it
              r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: Skipping unknown sub-parameter type 0x%X in RegulatoryCapabilities", subparam_info.type);
              //TODO Add a ctx handler to pass this event to the user
              break;
          }
          assert(subparam_info.len != 0xffff); // Make sure there are no unsupported TV params as that would break the parsing TODO
          suboffset += subparam_info.len;
        }
        break;
      }
      case R420_PARAM_TYPE_C1G2_LLRP_CAPABILITIES:
        // We can parse specific fields if needed
        break;
      default:
        // Unknown parameter type, we just skip it
        r420_logf(pCtx, "r420_process_get_reader_capabilities_response_msg: Skipping unknown parameter type 0x%X", param_info.type);
        //TODO Add a ctx handler to pass this event to the user
        break;
    }
    assert(param_info.len != 0xffff); // Make sure there are no unsupported TV params as that would break the parsing TODO
    offset += param_info.len;
  }
}

void r420_process_impinj_enable_extensions_response_msg(const r420_ctx_t *pCtx, const r420_msg_body_t *pBody) {
  // First 4 bytes - vendor ID
  // Next 1 byte - subtype
  // Next 4 bytes - TLV header for LLRPStatus parameter
  // Next 2 bytes - StatusCode
  // Next 2 bytes - Error Description ByteCount
  // ...
  assert(pBody->len >= 13); // We expect at least 13 bytes
  r420_msg_param_info_t param_info = r420_process_param(pCtx, pBody, 5);
  assert(param_info.type == R420_PARAM_TYPE_LLRP_STATUS);
  assert(param_info.len >= 4); // We expect at least 4 bytes for LLRPStatus
  uint16_t status_code = *(uint16_t *)(pBody->buf + param_info.value_offset);
  assert(status_code == 0); // We expect StatusCode=M_Success
  //uint16_t err_desc_len = *(uint16_t *)(pBody->buf + param_info.value_offset + 2);
  // We ignore the error description and error params for now
  // TODO Handle the error description and error params
  // TODO Parse other parameters in the ImpinjEnableExtensionsResponse message
}

void r420_process_get_reader_config_response_msg(const r420_ctx_t *pCtx, const r420_msg_body_t *pBody) {
  // First 4 bytes - TLV header for LLRPStatus parameter
  // Next 2 bytes - StatusCode
  // Next 2 bytes - Error Description ByteCount
  // ...
  assert(pBody->len >= 8); // We expect at least 8 bytes
  r420_msg_param_info_t param_info = r420_process_param(pCtx, pBody, 0);
  assert(param_info.type == R420_PARAM_TYPE_LLRP_STATUS);
  assert(param_info.len >= 4); // We expect at least 4 bytes for LLRPStatus
  uint16_t status_code = *(uint16_t *)(pBody->buf + param_info.value_offset);
  assert(status_code == 0); // We expect StatusCode=M
  //uint16_t err_desc_len = *(uint16_t *)(pBody->buf + param_info.value_offset + 2);
  // We ignore the error description and error params for now
  // TODO Handle the error description and error params
  // TODO Parse other parameters in the GetReaderConfigResponse message
}

void r420_process_set_reader_config_response_msg(const r420_ctx_t *pCtx, const r420_msg_body_t *pBody) {
  // First 4 bytes - TLV header for LLRPStatus parameter
  // Next 2 bytes - StatusCode
  // Next 2 bytes - Error Description ByteCount
  // ...
  assert(pBody->len >= 8); // We expect at least 8 bytes
  r420_msg_param_info_t param_info = r420_process_param(pCtx, pBody, 0);
  assert(param_info.type == R420_PARAM_TYPE_LLRP_STATUS);
  assert(param_info.len >= 4); // We expect at least 4 bytes for LLRPStatus
  uint16_t status_code = *(uint16_t *)(pBody->buf + param_info.value_offset);
  assert(status_code == 0); // We expect StatusCode=M_Success
  //uint16_t err_desc_len = *(uint16_t *)(pBody->buf + param_info.value_offset + 2);
  // We ignore the error description and error params for now
  // TODO Handle the error description and error params
}

void r420_process_get_rospecs_response_msg(r420_ctx_t *pCtx, const r420_msg_body_t *pBody) {
  // First 4 bytes - TLV header for LLRPStatus parameter
  // Next 2 bytes - StatusCode
  // Next 2 bytes - Error Description ByteCount
  // ...
  assert(pBody->len >= 8); // We expect at least 8 bytes
  r420_msg_param_info_t param_info = r420_process_param(pCtx, pBody, 0);
  assert(param_info.type == R420_PARAM_TYPE_LLRP_STATUS);
  assert(param_info.len >= 4); // We expect at least 4 bytes for LLRPStatus
  uint16_t status_code = *(uint16_t *)(pBody->buf + param_info.value_offset);
  assert(status_code == 0); // We expect StatusCode=M_Success
  //uint16_t err_desc_len = *(uint16_t *)(pBody->buf + param_info.value_offset + 2);
  // We ignore the error description and error params for now
  // TODO Handle the error description and error params
  // Check if there are any ROSpec parameters present
  size_t offset = param_info.value_offset + param_info.len - sizeof(r420_msg_body_param_tlv_hdr_t);
  if (pBody->len > offset) {
    // There are ROSpec parameters present
    r420_logf(pCtx, "ROSPECS already present in reader.");
    pCtx->rospec_added = 1;
    // Check CurrentState of the first ROSpec
    r420_msg_param_info_t rospec_param_info = r420_process_param(pCtx, pBody, offset);
    assert(rospec_param_info.type == R420_PARAM_TYPE_ROSPEC);
    assert(rospec_param_info.len >= 12);
    uint8_t current_state = *(uint8_t *)(pBody->buf + rospec_param_info.value_offset + 5); // CurrentState is at offset 5 within the ROSpec parameter value
    if (current_state == 0) { //disabled state
      pCtx->rospec_enabled = 0;
      pCtx->rospec_started = 0;
      r420_logf(pCtx, "ROSPEC is in DISABLED state.");
    } else if (current_state == 1) { //inactive state
      pCtx->rospec_enabled = 1;
      pCtx->rospec_started = 0;
      r420_logf(pCtx, "ROSPEC is in INACTIVE state.");
    } else if (current_state == 2) { //active state
      pCtx->rospec_enabled = 1;
      pCtx->rospec_started = 1;
      r420_logf(pCtx, "ROSPEC is in ACTIVE state.");
    } else {
      assert(0);
    }
  } else {
    r420_logf(pCtx, "No ROSPECS present in reader.");
  }
}

void r420_process_add_rospec_response_msg(r420_ctx_t *pCtx, const r420_msg_body_t *pBody) {
  // First 4 bytes - TLV header for LLRPStatus parameter
  // Next 2 bytes - StatusCode
  // Next 2 bytes - Error Description ByteCount
  // ...
  assert(pBody->len >= 8); // We expect at least 8 bytes
  r420_msg_param_info_t param_info = r420_process_param(pCtx, pBody, 0);
  assert(param_info.type == R420_PARAM_TYPE_LLRP_STATUS);
  assert(param_info.len >= 4); // We expect at least 4 bytes for LLRPStatus
  uint16_t status_code = *(uint16_t *)(pBody->buf + param_info.value_offset);
  assert(status_code == 0); // We expect StatusCode=M_Success
  pCtx->rospec_added = 1;
  //uint16_t err_desc_len = *(uint16_t *)(pBody->buf + param_info.value_offset + 2);
  // We ignore the error description and error params for now
  // TODO Handle the error description and error params
}

void r420_process_delete_rospec_response_msg(r420_ctx_t *pCtx, const r420_msg_body_t *pBody) {
  // First 4 bytes - TLV header for LLRPStatus parameter
  // Next 2 bytes - StatusCode
  // Next 2 bytes - Error Description ByteCount
  // ...
  assert(pBody->len >= 8); // We expect at least 8 bytes
  r420_msg_param_info_t param_info = r420_process_param(pCtx, pBody, 0);
  assert(param_info.type == R420_PARAM_TYPE_LLRP_STATUS);
  assert(param_info.len >= 4); // We expect at least 4 bytes for LLRPStatus
  uint16_t status_code = *(uint16_t *)(pBody->buf + param_info.value_offset);
  assert(status_code == 0); // We expect StatusCode=M_Success
  pCtx->rospec_added = 0;
  pCtx->rospec_enabled = 0;
  pCtx->rospec_started = 0;
  //uint16_t err_desc_len = *(uint16_t *)(pBody->buf + param_info.value_offset + 2);
  // We ignore the error description and error params for now
  // TODO Handle the error description and error params
}

void r420_process_enable_rospec_response_msg(r420_ctx_t *pCtx, const r420_msg_body_t *pBody) {
  // First 4 bytes - TLV header for LLRPStatus parameter
  // Next 2 bytes - StatusCode
  // Next 2 bytes - Error Description ByteCount
  // ...
  assert(pBody->len >= 8); // We expect at least 8 bytes
  r420_msg_param_info_t param_info = r420_process_param(pCtx, pBody, 0);
  assert(param_info.type == R420_PARAM_TYPE_LLRP_STATUS);
  assert(param_info.len >= 4); // We expect at least 4 bytes for LLRPStatus
  uint16_t status_code = *(uint16_t *)(pBody->buf + param_info.value_offset);
  assert(status_code == 0); // We expect StatusCode=M_Success
  //uint16_t err_desc_len = *(uint16_t *)(pBody->buf + param_info.value_offset + 2);
  // We ignore the error description and error params for now
  // TODO Handle the error description and error params
}

void r420_process_disable_rospec_response_msg(r420_ctx_t *pCtx, const r420_msg_body_t *pBody) {
  // First 4 bytes - TLV header for LLRPStatus parameter
  // Next 2 bytes - StatusCode
  // Next 2 bytes - Error Description ByteCount
  // ...
  assert(pBody->len >= 8); // We expect at least 8 bytes
  r420_msg_param_info_t param_info = r420_process_param(pCtx, pBody, 0);
  assert(param_info.type == R420_PARAM_TYPE_LLRP_STATUS);
  assert(param_info.len >= 4); // We expect at least 4 bytes for LLRPStatus
  uint16_t status_code = *(uint16_t *)(pBody->buf + param_info.value_offset);
  assert(status_code == 0); // We expect StatusCode=M_Success
  pCtx->rospec_enabled = 0;
  pCtx->rospec_started = 0;
  //uint16_t err_desc_len = *(uint16_t *)(pBody->buf + param_info.value_offset + 2);
  // We ignore the error description and error params for now
  // TODO Handle the error description and error params
}

void r420_process_start_rospec_response_msg(r420_ctx_t *pCtx, const r420_msg_body_t *pBody) {
  // First 4 bytes - TLV header for LLRPStatus parameter
  // Next 2 bytes - StatusCode
  // Next 2 bytes - Error Description ByteCount
  // ...
  assert(pBody->len >= 8); // We expect at least 8 bytes
  r420_msg_param_info_t param_info = r420_process_param(pCtx, pBody, 0);
  assert(param_info.type == R420_PARAM_TYPE_LLRP_STATUS);
  assert(param_info.len >= 4); // We expect at least 4 bytes for LLRPStatus
  uint16_t status_code = *(uint16_t *)(pBody->buf + param_info.value_offset);
  assert(status_code == 0); // We expect StatusCode=M_Success
  pCtx->rospec_started = 1;
  //uint16_t err_desc_len = *(uint16_t *)(pBody->buf + param_info.value_offset + 2);
  // We ignore the error description and error params for now
  // TODO Handle the error description and error params
}

void r420_process_stop_rospec_response_msg(r420_ctx_t *pCtx, const r420_msg_body_t *pBody) {
  // First 4 bytes - TLV header for LLRPStatus parameter
  // Next 2 bytes - StatusCode
  // Next 2 bytes - Error Description ByteCount
  // ...
  assert(pBody->len >= 8); // We expect at least 8 bytes
  r420_msg_param_info_t param_info = r420_process_param(pCtx, pBody, 0);
  assert(param_info.type == R420_PARAM_TYPE_LLRP_STATUS);
  assert(param_info.len >= 4); // We expect at least 4 bytes for LLRPStatus
  uint16_t status_code = *(uint16_t *)(pBody->buf + param_info.value_offset);
  assert(status_code == 0); // We expect StatusCode=M_Success
  pCtx->rospec_started = 0;
  //uint16_t err_desc_len = *(uint16_t *)(pBody->buf + param_info.value_offset + 2);
  // We ignore the error description and error params for now
  // TODO Handle the error description and error params
}

void r420_process_ro_access_report_msg(const r420_ctx_t *pCtx, const r420_msg_body_t *pBody) {
  assert(pBody->len >= 0);
  r420_logf(pCtx, "r420_process_ro_access_report_msg: Received ROAccessReport message of length %zu bytes", pBody->len);
  
  size_t offset = 0;
  while (offset < pBody->len) {
    r420_msg_param_info_t param_info = r420_process_param(pCtx, pBody, offset);
    switch(param_info.type) {
      case R420_PARAM_TYPE_TAG_REPORT_DATA:
        assert(param_info.len >= 0);

        uint8_t peak_rssi;
        uint8_t epc96[12] = {0};
        uint16_t antenna_id;
        uint16_t rf_phase_angle;
        int16_t rf_doppler_frequency;
        int16_t peak_rssi_16bit;
        uint64_t first_seen_timestamp_utc_microseconds;
        uint64_t last_seen_timestamp_utc_microseconds;
        uint16_t tag_seen_count;
        uint16_t channel_index;

        int peak_rssi_is_set = 0;
        int epc96_is_set = 0;
        int antenna_id_is_set = 0;
        int rf_phase_angle_is_set = 0;
        int rf_doppler_frequency_is_set = 0;
        int peak_rssi_16bit_is_set = 0;
        int first_seen_timestamp_utc_microseconds_is_set = 0;
        int last_seen_timestamp_utc_microseconds_is_set = 0;
        int tag_seen_count_is_set = 0;
        int channel_index_is_set = 0;

        size_t suboffset = 0;

        while (param_info.value_offset + suboffset < offset + param_info.len) {
          r420_msg_param_info_t subparam_info = r420_process_param(pCtx, pBody, param_info.value_offset + suboffset);
          switch(subparam_info.type) {
            case R420_PARAM_TYPE_EPC_96:
              assert(epc96_is_set == 0);
              assert(NULL != memcpy(epc96, pBody->buf + subparam_info.value_offset, sizeof(epc96)));
              epc96_is_set = 1;
              break;
            case R420_PARAM_TYPE_ANTENNA_ID:
              assert(antenna_id_is_set == 0);
              antenna_id = ntohs(*(uint16_t*)(pBody->buf + subparam_info.value_offset));
              antenna_id_is_set = 1;
              break;
            case R420_PARAM_TYPE_PEAK_RSSI:
              assert(peak_rssi_is_set == 0);
              peak_rssi = *(uint8_t*)(pBody->buf + subparam_info.value_offset);
              peak_rssi_is_set = 1;
              break;
            case R420_PARAM_TYPE_FIRST_SEEN_TIMESTAMP_UTC:
              assert(first_seen_timestamp_utc_microseconds_is_set == 0);
              first_seen_timestamp_utc_microseconds = ntohll(*(uint64_t *)(pBody->buf + subparam_info.value_offset));
              first_seen_timestamp_utc_microseconds_is_set = 1;
              break;
            case R420_PARAM_TYPE_LAST_SEEN_TIMESTAMP_UTC:
              assert(last_seen_timestamp_utc_microseconds_is_set == 0);
              last_seen_timestamp_utc_microseconds = ntohll(*(uint64_t *)(pBody->buf + subparam_info.value_offset));
              last_seen_timestamp_utc_microseconds_is_set = 1;
              break;
            case R420_PARAM_TYPE_TAG_SEEN_COUNT:
              assert(tag_seen_count_is_set == 0);
              tag_seen_count = ntohs(*(uint16_t *)(pBody->buf + subparam_info.value_offset));
              tag_seen_count_is_set = 1;
              break;
            case R420_PARAM_TYPE_CHANNEL_INDEX:
              assert(channel_index_is_set == 0);
              channel_index = ntohs(*(uint16_t *)(pBody->buf + subparam_info.value_offset));
              channel_index_is_set = 1;
              break;
            case R420_PARAM_TYPE_CUSTOM_PARAMETER:
              uint32_t vendor_id = ntohl(*(uint32_t *)(pBody->buf + subparam_info.value_offset));
              uint32_t subtype = ntohl(*(uint32_t *)(pBody->buf + subparam_info.value_offset + sizeof(vendor_id)));
              switch (vendor_id) {
                case IMPINJ_VENDOR_ID:
                  switch (subtype) {
                    case R420_CUSTOM_PARAMETER_SUBTYPE_IMPINJ_RF_PHASE_ANGLE:
                      assert(rf_phase_angle_is_set == 0);
                      rf_phase_angle = ntohs(*(uint16_t *)(pBody->buf + subparam_info.value_offset + sizeof(vendor_id) + sizeof(subtype)));
                      rf_phase_angle_is_set = 1;
                      break;
                    case R420_CUSTOM_PARAMETER_SUBTYPE_IMPINJ_RF_DOPPLER_FREQUENCY:
                      assert(rf_doppler_frequency_is_set == 0);                      
                      rf_doppler_frequency = ntohs(*(int16_t *)(pBody->buf + subparam_info.value_offset + sizeof(vendor_id) + sizeof(subtype)));
                      rf_doppler_frequency_is_set = 1;
                      break;
                    case R420_CUSTOM_PARAMETER_SUBTYPE_IMPINJ_PEAK_RSSI:
                      assert(peak_rssi_16bit_is_set == 0);                      
                      peak_rssi_16bit = ntohs(*(int16_t *)(pBody->buf + subparam_info.value_offset + sizeof(vendor_id) + sizeof(subtype)));
                      peak_rssi_16bit_is_set = 1;
                      break;
                    default:
                      //TODO
                      break;
                  }
                  break;
                default:
                  //TODO
                  break;
              }
              break;
            default:
              //TODO
              break;
          }
          assert(subparam_info.len != 0xffff); // Make sure there are no unsupported TV params as that would break the parsing TODO
          suboffset += subparam_info.len;
        }
        
        if (peak_rssi_is_set && epc96_is_set && antenna_id_is_set && rf_phase_angle_is_set && rf_doppler_frequency_is_set &  peak_rssi_16bit_is_set && first_seen_timestamp_utc_microseconds_is_set && last_seen_timestamp_utc_microseconds_is_set && tag_seen_count_is_set && channel_index_is_set) {
          r420_logf(pCtx, "Tag EPC: %02X%02X%02X%02X%02X%02X%02X%02X%02X%02X%02X%02X, Antenna ID: %u, Peak RSSI: %d, RF Phase Angle: %d, RF Doppler Frequency: %d, 16-bit peak rssi: %d, First Seen Timestamp (UTC): %llu us, Last Seen Timestamp (UTC): %llu us, Tag Seen Count: %u, Channel Index: %u",
            epc96[0], epc96[1], epc96[2], epc96[3], epc96[4], epc96[5],
            epc96[6], epc96[7], epc96[8], epc96[9], epc96[10], epc96[11],
            antenna_id, peak_rssi, rf_phase_angle, rf_doppler_frequency, peak_rssi_16bit,
            (unsigned long long)first_seen_timestamp_utc_microseconds,
            (unsigned long long)last_seen_timestamp_utc_microseconds,
            tag_seen_count, channel_index);
          //TODO pass to measurement-dspecific user handler
        } else {
          r420_logf(pCtx, "Incomplete tag report data received.");
          //TODO pass to user handler
          if (!peak_rssi_is_set) {
            r420_logf(pCtx, "  Missing Peak RSSI");
          }
          if (!epc96_is_set) {
            r420_logf(pCtx, "  Missing EPC-96");
          }
          if (!antenna_id_is_set) {
            r420_logf(pCtx, "  Missing Antenna ID");
          }
          if (!rf_phase_angle_is_set) {
            r420_logf(pCtx, "  Missing RF Phase Angle");
          }
          if (!rf_doppler_frequency_is_set) {
            r420_logf(pCtx, "  Missing RF Doppler Frequency");
          }
          if (!peak_rssi_16bit_is_set) {
            r420_logf(pCtx, "  Missing 16-bit Peak RSSI");
          }
          if (!first_seen_timestamp_utc_microseconds_is_set) {
            r420_logf(pCtx, "  Missing First Seen Timestamp (UTC)");
          }
          if (!last_seen_timestamp_utc_microseconds_is_set) {
            r420_logf(pCtx, "  Missing Last Seen Timestamp (UTC)");
          }
          if (!tag_seen_count_is_set) {
            r420_logf(pCtx, "  Missing Tag Seen Count");
          }
          if (!channel_index_is_set) {
            r420_logf(pCtx, "  Missing Channel Index");
          }
        }
        
        break;
      case R420_PARAM_TYPE_RF_SURVEY_REPORT_DATA:
        break;
      default:
        //TODO
        break;
    }
    assert(param_info.len != 0xffff); // Make sure there are no unsupported TV params as that would break the parsing TODO
    offset += param_info.len;
  }
}

void r420_send_message(r420_ctx_t *pCtx, const r420_msg_hdr_t *pHdr, const r420_msg_body_t *pBody) {
  // Send header
  ssize_t remaining_snd_bytes = sizeof(*pHdr);
  while (remaining_snd_bytes > 0) {
    ssize_t n = write(pCtx->fd, ((const uint8_t *)pHdr) + (sizeof(*pHdr) - remaining_snd_bytes), remaining_snd_bytes);
    assert(n > 0);
    remaining_snd_bytes -= n;
  }
  // Send body
  remaining_snd_bytes = pBody->len;
  while (remaining_snd_bytes > 0) {
    ssize_t n = write(pCtx->fd, pBody->buf + (pBody->len - remaining_snd_bytes), remaining_snd_bytes);
    assert(n > 0);
    remaining_snd_bytes -= n;
  }
  pCtx->next_tx_msg_id++; // Increment message ID for next message
}

void r420_send_get_supported_version_msg(r420_ctx_t *pCtx) {
  r420_msg_hdr_t hdr = {
    /* version = ctx->llrp_version, message type = R420_MSG_TYPE_GET_SUPPORTED_VERSION */
    .attrs = htons((pCtx->llrp_version << 10) | R420_MSG_TYPE_GET_SUPPORTED_VERSION),
    .message_length = htonl(sizeof(r420_msg_hdr_t)), // No body
    .message_id = htonl(pCtx->next_tx_msg_id)
  };
  assert(R420_MSG_HDR_VERSION(hdr) == pCtx->llrp_version);
  assert(R420_MSG_HDR_MESSAGE_TYPE(hdr) == R420_MSG_TYPE_GET_SUPPORTED_VERSION);
  assert(R420_MSG_HDR_MSG_LENGTH(hdr) == sizeof(r420_msg_hdr_t));
  assert(R420_MSG_HDR_MSG_ID(hdr) == pCtx->next_tx_msg_id);

  r420_msg_body_t body = { .buf = {0}, .len = 0 };
  r420_send_message(pCtx, &hdr, &body);
}

#define R420_GET_READER_CAPABILITIES_REQUESTED_DATA_ALL 0
#define R420_GET_READER_CAPABILITIES_REQUESTED_DATA_GENERAL_DEVICE_CAPABILITIES 1
#define R420_GET_READER_CAPABILITIES_REQUESTED_DATA_LLRP_CAPABILITIES 2
#define R420_GET_READER_CAPABILITIES_REQUESTED_DATA_REGULATORY_CAPABILITIES 3
#define R420_GET_READER_CAPABILITIES_REQUESTED_DATA_AIR_PROTOCOL_CAPABILITIES 4

void r420_send_get_reader_capabilities_msg(r420_ctx_t *pCtx) {
  uint8_t requested_data = R420_GET_READER_CAPABILITIES_REQUESTED_DATA_ALL;

  //ImpinjRequestedData
  uint32_t impinj_requested_data_vendor_id = IMPINJ_VENDOR_ID;
  uint32_t impinj_requested_data_subtype = R420_CUSTOM_PARAMETER_SUBTYPE_IMPINJ_REQUESTED_DATA;
  uint32_t impinj_requested_data = 1000; // All_Capabilities

  r420_msg_body_param_tlv_hdr_t impinj_requested_data_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_CUSTOM_PARAMETER), // reserved=0, type=CustomParameter
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(impinj_requested_data_vendor_id) + sizeof(impinj_requested_data_subtype) + sizeof(impinj_requested_data))
  };

  /* Construct the message body */
  r420_msg_body_t body = { .buf = {0}, .len = sizeof(requested_data) + ntohs(impinj_requested_data_param_hdr.param_len) };
  size_t offset = 0;
  body.buf[offset] = requested_data;
  offset += sizeof(requested_data);
  // Copy ImpinjRequestedData tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = impinj_requested_data_param_hdr;
  offset += sizeof(impinj_requested_data_param_hdr);
  // Copy ImpinjRequestedData vendor ID
  *(uint32_t *)(body.buf + offset) = htonl(impinj_requested_data_vendor_id);
  offset += sizeof(impinj_requested_data_vendor_id);
  // Copy ImpinjRequestedData subtype
  *(uint32_t *)(body.buf + offset) = htonl(impinj_requested_data_subtype);
  offset += sizeof(impinj_requested_data_subtype);
  // Copy ImpinjRequestedData RequestedData
  *(uint32_t *)(body.buf + offset) = htonl(impinj_requested_data);
  offset += sizeof(impinj_requested_data);

  assert(offset == body.len);
  r420_msg_hdr_t hdr = {
    /* version = ctx->llrp_version, message type = R420_MSG_TYPE_GET_READER_CAPABILITIES */
    .attrs = htons((pCtx->llrp_version << 10) | R420_MSG_TYPE_GET_READER_CAPABILITIES),
    .message_length = htonl(sizeof(r420_msg_hdr_t) + body.len),
    .message_id = htonl(pCtx->next_tx_msg_id)
  };
  assert(R420_MSG_HDR_VERSION(hdr) == pCtx->llrp_version);
  assert(R420_MSG_HDR_MESSAGE_TYPE(hdr) == R420_MSG_TYPE_GET_READER_CAPABILITIES);
  assert(R420_MSG_HDR_MSG_LENGTH(hdr) == sizeof(r420_msg_hdr_t) + body.len);
  assert(R420_MSG_HDR_MSG_ID(hdr) == pCtx->next_tx_msg_id);

  r420_send_message(pCtx, &hdr, &body);
}

void r420_send_impinj_enable_extensions_msg(r420_ctx_t* pCtx) {
  uint32_t vendor_id = IMPINJ_VENDOR_ID;
  uint8_t subtype = R420_CUSTOM_MESSAGE_SUBTYPE_IMPINJ_ENABLE_EXTENSIONS;
  uint32_t reserved = 0;
  r420_msg_body_t body = { .buf = {0}, .len = sizeof(vendor_id) + sizeof(subtype) + sizeof(reserved) };
  size_t offset = 0;
  *(uint32_t *)(body.buf + offset) = htonl(vendor_id);
  offset += sizeof(vendor_id);
  *(uint8_t *)(body.buf + offset) = subtype;
  offset += sizeof(subtype);
  *(uint32_t *)(body.buf + offset) = htonl(reserved);
  offset += sizeof(reserved);
  assert(offset == body.len);
  r420_msg_hdr_t hdr = {
    /* version = ctx->llrp_version, message type = R420_MSG_TYPE_CUSTOM_MESSAGE */
    .attrs = htons((pCtx->llrp_version << 10) | R420_MSG_TYPE_CUSTOM_MESSAGE),
    .message_length = htonl(sizeof(r420_msg_hdr_t) + body.len),
    .message_id = htonl(pCtx->next_tx_msg_id)
  };
  assert(R420_MSG_HDR_VERSION(hdr) == pCtx->llrp_version);
  assert(R420_MSG_HDR_MESSAGE_TYPE(hdr) == R420_MSG_TYPE_CUSTOM_MESSAGE);
  assert(R420_MSG_HDR_MSG_LENGTH(hdr) == sizeof(r420_msg_hdr_t) + body.len);
  assert(R420_MSG_HDR_MSG_ID(hdr) == pCtx->next_tx_msg_id);
  r420_send_message(pCtx, &hdr, &body);
}

void r420_send_get_reader_config_msg(r420_ctx_t* pCtx) {
  uint16_t antenna_id = 0; // 0 means all antennas
  uint8_t requested_data = 0; // 0 means all config data
  uint16_t gpi_port_num = 0; // 0 means all GPI ports
  uint16_t gpo_port_num = 0; // 0 means all GPO ports

  //ImpinjRequestedData
  uint32_t impinj_requested_data_vendor_id = IMPINJ_VENDOR_ID;
  uint32_t impinj_requested_data_subtype = R420_CUSTOM_PARAMETER_SUBTYPE_IMPINJ_REQUESTED_DATA;
  uint32_t impinj_requested_data = 2000; // All_Configuration

  r420_msg_body_param_tlv_hdr_t impinj_requested_data_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_CUSTOM_PARAMETER), // reserved=0, type=CustomParameter
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(impinj_requested_data_vendor_id) + sizeof(impinj_requested_data_subtype) + sizeof(impinj_requested_data))
  };

  /* Construct the message body */
  r420_msg_body_t body = { .buf = {0}, .len = sizeof(antenna_id) + sizeof(requested_data) + sizeof(gpi_port_num) + sizeof(gpo_port_num) + ntohs(impinj_requested_data_param_hdr.param_len) };
  size_t offset = 0;
  // Copy antenna ID
  *(uint16_t *)(body.buf + offset) = htons(antenna_id);
  offset += sizeof(antenna_id);
  body.buf[offset] = requested_data;
  offset += sizeof(requested_data);
  // Copy GPIPortNum
  *(uint16_t *)(body.buf + offset) = htons(gpi_port_num);
  offset += sizeof(gpi_port_num);
  // Copy GPOPortNum
  *(uint16_t *)(body.buf + offset) = htons(gpo_port_num);
  offset += sizeof(gpo_port_num);
  // Copy ImpinjRequestedData tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = impinj_requested_data_param_hdr;
  offset += sizeof(impinj_requested_data_param_hdr);
  // Copy ImpinjRequestedData vendor ID
  *(uint32_t *)(body.buf + offset) = htonl(impinj_requested_data_vendor_id);
  offset += sizeof(impinj_requested_data_vendor_id);
  // Copy ImpinjRequestedData subtype
  *(uint32_t *)(body.buf + offset) = htonl(impinj_requested_data_subtype);
  offset += sizeof(impinj_requested_data_subtype);
  // Copy ImpinjRequestedData RequestedData
  *(uint32_t *)(body.buf + offset) = htonl(impinj_requested_data);
  offset += sizeof(impinj_requested_data);

  assert(offset == body.len);
  r420_msg_hdr_t hdr = {
    /* version = ctx->llrp_version, message type = R420_MSG_TYPE_GET_READER_CONFIG */
    .attrs = htons((pCtx->llrp_version << 10) | R420_MSG_TYPE_GET_READER_CONFIG),
    .message_length = htonl(sizeof(r420_msg_hdr_t) + body.len),
    .message_id = htonl(pCtx->next_tx_msg_id)
  };
  assert(R420_MSG_HDR_VERSION(hdr) == pCtx->llrp_version);
  assert(R420_MSG_HDR_MESSAGE_TYPE(hdr) == R420_MSG_TYPE_GET_READER_CONFIG);
  assert(R420_MSG_HDR_MSG_LENGTH(hdr) == sizeof(r420_msg_hdr_t) + body.len);
  assert(R420_MSG_HDR_MSG_ID(hdr) == pCtx->next_tx_msg_id);

  r420_send_message(pCtx, &hdr, &body);
}

void r420_send_set_reader_config_msg(r420_ctx_t* pCtx) {
  uint8_t reset_to_factory_default = 0; // don't reset
  uint8_t reserved = 0;
  uint8_t flags = reset_to_factory_default | (reserved << 1);

  uint8_t ro_report_trigger = 2; // Upon N TagReportData Parameters or End of ROSpec
  uint16_t n = 1; // N=1

  // TagReportContentSelector
  uint16_t enable_rospec_id = 0;
  uint16_t enable_spec_index = 0;
  uint16_t enable_inventory_parameter_spec_id = 0;
  uint16_t enable_antenna_id = 1;
  uint16_t enable_channel_index = 1;
  uint16_t enable_peak_rssi = 1;
  uint16_t enable_first_seen_timestamp = 1;
  uint16_t enable_last_seen_timestamp = 1;
  uint16_t enable_tag_seen_count = 1;
  uint16_t enable_access_spec_id = 0;
  uint16_t tag_report_content_selector_reserved = 0;

  uint16_t tag_report_content_selector_flags = tag_report_content_selector_reserved |
                                                (enable_access_spec_id << 6) |
                                                (enable_tag_seen_count << 7) |
                                                (enable_last_seen_timestamp << 8) |
                                                (enable_first_seen_timestamp << 9) |
                                                (enable_peak_rssi << 10) |
                                                (enable_channel_index << 11) |
                                                (enable_antenna_id << 12) |
                                                (enable_inventory_parameter_spec_id << 13) |
                                                (enable_spec_index << 14) |
                                                (enable_rospec_id << 15);

  // C1G2EPCMemorySelector
  uint8_t enable_crc = 0;
  uint8_t enable_pc_bits = 0;
  uint8_t c1g2_epc_memory_selector_reserved = 0;

  uint8_t c1g2_epc_memory_selector_flags = enable_crc |
                                           (enable_pc_bits << 1) |
                                           (c1g2_epc_memory_selector_reserved << 2);

  //ImpinjTagReportContentSelector
  uint32_t impinj_tag_report_content_selector_vendor_id = IMPINJ_VENDOR_ID;
  uint32_t impinj_tag_report_content_selector_subtype = R420_CUSTOM_PARAMETER_SUBTYPE_IMPINJ_TAG_REPORT_CONTENT_SELECTOR;

  //ImpinjEnableRFPhaseAngle
  uint32_t impinj_enable_rf_phase_angle_vendor_id = IMPINJ_VENDOR_ID;
  uint32_t impinj_enable_rf_phase_angle_subtype = R420_CUSTOM_PARAMETER_SUBTYPE_IMPINJ_ENABLE_RF_PHASE_ANGLE;
  uint16_t rf_phase_angle_mode = 1; // Enable

  //ImpinjEnablePeakRSSI
  uint32_t impinj_enable_peak_rssi_vendor_id = IMPINJ_VENDOR_ID;
  uint32_t impinj_enable_peak_rssi_subtype = R420_CUSTOM_PARAMETER_SUBTYPE_IMPINJ_ENABLE_PEAK_RSSI;
  uint16_t peak_rssi_mode = 1; // Enable

  //ImpinjEnableRFDopplerFrequency
  uint32_t impinj_enable_rf_doppler_frequency_vendor_id = IMPINJ_VENDOR_ID;
  uint32_t impinj_enable_rf_doppler_frequency_subtype = R420_CUSTOM_PARAMETER_SUBTYPE_IMPINJ_ENABLE_RF_DOPPLER_FREQUENCY;
  uint16_t rf_doppler_frequency_mode = 1; // Enable

  // Construct the headers for body parameters
  r420_msg_body_param_tlv_hdr_t impinj_enable_rf_phase_angle_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_CUSTOM_PARAMETER), // reserved=0, type=CustomParameter
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(impinj_enable_rf_phase_angle_vendor_id) + sizeof(impinj_enable_rf_phase_angle_subtype) + sizeof(rf_phase_angle_mode))
  };

  r420_msg_body_param_tlv_hdr_t impinj_enable_rf_doppler_frequency_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_CUSTOM_PARAMETER), // reserved=0, type=CustomParameter
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(impinj_enable_rf_doppler_frequency_vendor_id) + sizeof(impinj_enable_rf_doppler_frequency_subtype) + sizeof(rf_doppler_frequency_mode))
  };

  r420_msg_body_param_tlv_hdr_t impinj_enable_peak_rssi_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_CUSTOM_PARAMETER), // reserved=0, type=CustomParameter
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(impinj_enable_peak_rssi_vendor_id) + sizeof(impinj_enable_peak_rssi_subtype) + sizeof(peak_rssi_mode))
  };

  r420_msg_body_param_tlv_hdr_t impinj_tag_report_content_selector_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_CUSTOM_PARAMETER), // reserved=0, type=CustomParameter
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(impinj_tag_report_content_selector_vendor_id) + sizeof(impinj_tag_report_content_selector_subtype) + ntohs(impinj_enable_rf_phase_angle_param_hdr.param_len) + ntohs(impinj_enable_peak_rssi_param_hdr.param_len) + ntohs(impinj_enable_rf_doppler_frequency_param_hdr.param_len))
  };

  r420_msg_body_param_tlv_hdr_t c1g2_epc_memory_selector_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_C1G2_MEMORY_SELECTOR), // reserved=0, type=C1G2EPCMemorySelector
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(c1g2_epc_memory_selector_flags))
  };

  r420_msg_body_param_tlv_hdr_t tag_report_content_selector_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_TAG_REPORT_CONTENT_SELECTOR), // reserved=0, type=TagReportContentSelector
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(tag_report_content_selector_flags) + ntohs(c1g2_epc_memory_selector_param_hdr.param_len))
  };

  r420_msg_body_param_tlv_hdr_t ro_report_spec_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_RO_REPORT_SPEC), // reserved=0, type=ROReportSpec
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(ro_report_trigger) + sizeof(n) + ntohs(tag_report_content_selector_param_hdr.param_len) + ntohs(impinj_tag_report_content_selector_param_hdr.param_len))
  };

  //ImpinjReportBufferConfiguration
  uint32_t impinj_report_buffer_configuration_vendor_id = IMPINJ_VENDOR_ID;
  uint32_t impinj_report_buffer_configuration_subtype = R420_CUSTOM_PARAMETER_SUBTYPE_IMPINJ_REPORT_BUFFER_CONFIGURATION;
  uint16_t report_buffer_mode = 1; // Low_Latency

  r420_msg_body_param_tlv_hdr_t impinj_report_buffer_configuration_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_CUSTOM_PARAMETER), // reserved=0, type=CustomParameter
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(impinj_report_buffer_configuration_vendor_id) + sizeof(impinj_report_buffer_configuration_subtype) + sizeof(report_buffer_mode))
  };

  /* Construct the message body */
  r420_msg_body_t body = { .buf = {0}, .len = sizeof(flags) + ntohs(ro_report_spec_param_hdr.param_len) + ntohs(impinj_report_buffer_configuration_param_hdr.param_len) };
  size_t offset = 0;
  body.buf[offset] = flags;
  offset += sizeof(flags);
  // Copy ROReportSpec tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = ro_report_spec_param_hdr;
  offset += sizeof(r420_msg_body_param_tlv_hdr_t);
  // Copy ROReportTrigger
  body.buf[offset] = ro_report_trigger;
  offset += sizeof(ro_report_trigger);
  // Copy N
  *(uint16_t *)(body.buf + offset) = htons(n);
  offset += sizeof(n);
  // Copy TagReportContentSelector tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = tag_report_content_selector_param_hdr;
  offset += sizeof(r420_msg_body_param_tlv_hdr_t);
  // Copy TagReportContentSelector flags
  *(uint16_t *)(body.buf + offset) = htons(tag_report_content_selector_flags);
  offset += sizeof(tag_report_content_selector_flags);
  // Copy C1G2EPCMemorySelector tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = c1g2_epc_memory_selector_param_hdr;
  offset += sizeof(r420_msg_body_param_tlv_hdr_t);
  // Copy C1G2EPCMemorySelector flags
  *(uint8_t *)(body.buf + offset) = c1g2_epc_memory_selector_flags;
  offset += sizeof(c1g2_epc_memory_selector_flags);
  // Copy ImpinjTagReportContentSelector tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = impinj_tag_report_content_selector_param_hdr;
  offset += sizeof(r420_msg_body_param_tlv_hdr_t);
  // Copy ImpinjTagReportContentSelector vendor ID
  *(uint32_t *)(body.buf + offset) = htonl(impinj_tag_report_content_selector_vendor_id);
  offset += sizeof(impinj_tag_report_content_selector_vendor_id);
  // Copy ImpinjTagReportContentSelector subtype
  *(uint32_t *)(body.buf + offset) = htonl(impinj_tag_report_content_selector_subtype);
  offset += sizeof(impinj_tag_report_content_selector_subtype);
  // Copy ImpinjEnableRFPhaseAngle tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = impinj_enable_rf_phase_angle_param_hdr;
  offset += sizeof(r420_msg_body_param_tlv_hdr_t);
  // Copy ImpinjEnableRFPhaseAngle vendor ID
  *(uint32_t *)(body.buf + offset) = htonl(impinj_enable_rf_phase_angle_vendor_id);
  offset += sizeof(impinj_enable_rf_phase_angle_vendor_id);
  // Copy ImpinjEnableRFPhaseAngle subtype
  *(uint32_t *)(body.buf + offset) = htonl(impinj_enable_rf_phase_angle_subtype);
  offset += sizeof(impinj_enable_rf_phase_angle_subtype);
  // Copy RFPhaseAngleMode
  *(uint16_t *)(body.buf + offset) = htons(rf_phase_angle_mode);
  offset += sizeof(rf_phase_angle_mode);
  // Copy ImpinjEnablePeakRSSI tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = impinj_enable_peak_rssi_param_hdr;
  offset += sizeof(r420_msg_body_param_tlv_hdr_t);
  // Copy ImpinjEnablePeakRSSI vendor ID
  *(uint32_t *)(body.buf + offset) = htonl(impinj_enable_peak_rssi_vendor_id);
  offset += sizeof(impinj_enable_peak_rssi_vendor_id);
  // Copy ImpinjEnablePeakRSSI subtype
  *(uint32_t *)(body.buf + offset) = htonl(impinj_enable_peak_rssi_subtype);
  offset += sizeof(impinj_enable_peak_rssi_subtype);
  // Copy PeakRSSIMode
  *(uint16_t *)(body.buf + offset) = htons(peak_rssi_mode);
  offset += sizeof(peak_rssi_mode);
  // Copy ImpinjEnableRFDopplerFrequency tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = impinj_enable_rf_doppler_frequency_param_hdr;
  offset += sizeof(r420_msg_body_param_tlv_hdr_t);
  // Copy ImpinjEnableRFDopplerFrequency vendor ID
  *(uint32_t *)(body.buf + offset) = htonl(impinj_enable_rf_doppler_frequency_vendor_id);
  offset += sizeof(impinj_enable_rf_doppler_frequency_vendor_id);
  // Copy ImpinjEnableRFDopplerFrequency subtype
  *(uint32_t *)(body.buf + offset) = htonl(impinj_enable_rf_doppler_frequency_subtype);
  offset += sizeof(impinj_enable_rf_doppler_frequency_subtype);
  // Copy RFDopplerFrequencyMode
  *(uint16_t *)(body.buf + offset) = htons(rf_doppler_frequency_mode);
  offset += sizeof(rf_doppler_frequency_mode);
  // Copy ImpinjReportBufferConfiguration tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = impinj_report_buffer_configuration_param_hdr;
  offset += sizeof(r420_msg_body_param_tlv_hdr_t);
  // Copy ImpinjReportBufferConfiguration vendor ID
  *(uint32_t *)(body.buf + offset) = htonl(impinj_report_buffer_configuration_vendor_id);
  offset += sizeof(impinj_report_buffer_configuration_vendor_id);
  // Copy ImpinjReportBufferConfiguration subtype
  *(uint32_t *)(body.buf + offset) = htonl(impinj_report_buffer_configuration_subtype);
  offset += sizeof(impinj_report_buffer_configuration_subtype);
  // Copy ReportBufferMode
  *(uint16_t *)(body.buf + offset) = htons(report_buffer_mode);
  offset += sizeof(report_buffer_mode);

  assert(offset == body.len);
  r420_msg_hdr_t hdr = {
    /* version = ctx->llrp_version, message type = R420_MSG_TYPE_SET_READER_CONFIG */
    .attrs = htons((pCtx->llrp_version << 10) | R420_MSG_TYPE_SET_READER_CONFIG),
    .message_length = htonl(sizeof(r420_msg_hdr_t) + body.len),
    .message_id = htonl(pCtx->next_tx_msg_id)
  };
  assert(R420_MSG_HDR_VERSION(hdr) == pCtx->llrp_version);
  assert(R420_MSG_HDR_MESSAGE_TYPE(hdr) == R420_MSG_TYPE_SET_READER_CONFIG);
  assert(R420_MSG_HDR_MSG_LENGTH(hdr) == sizeof(r420_msg_hdr_t) + body.len);
  assert(R420_MSG_HDR_MSG_ID(hdr) == pCtx->next_tx_msg_id);
  r420_send_message(pCtx, &hdr, &body);
}

void r420_send_get_rospecs_msg(r420_ctx_t* pCtx) {
  r420_msg_body_t body = { .buf = {0}, .len = 0 }; // No body

  r420_msg_hdr_t hdr = {
    /* version = ctx->llrp_version, message type = R420_MSG_TYPE_GET_ROSPECS */
    .attrs = htons((pCtx->llrp_version << 10) | R420_MSG_TYPE_GET_ROSPECS),
    .message_length = htonl(sizeof(r420_msg_hdr_t) + body.len),
    .message_id = htonl(pCtx->next_tx_msg_id)
  };
  assert(R420_MSG_HDR_VERSION(hdr) == pCtx->llrp_version);
  assert(R420_MSG_HDR_MESSAGE_TYPE(hdr) == R420_MSG_TYPE_GET_ROSPECS);
  assert(R420_MSG_HDR_MSG_LENGTH(hdr) == sizeof(r420_msg_hdr_t) + body.len);
  assert(R420_MSG_HDR_MSG_ID(hdr) == pCtx->next_tx_msg_id);

  r420_send_message(pCtx, &hdr, &body);
}

// typedef struct r420_rospec {
//   // TODO Define the structure of a ROSpec
//   uint32_t rospec_id;
// } r420_rospec_t;

//void r420_send_add_rospec_msg(r420_ctx_t* pCtx, const r420_rospec_t* pRospec) {
void r420_send_add_rospec_msg(r420_ctx_t* pCtx) {
  uint32_t rospec_id = 1;
  uint8_t priority = 0;
  uint8_t current_state = 0; // Disabled (required by standard)

  uint8_t rospec_start_trigger_type = 0; // Null - the only way to start the ROSpec is via a START_ROSPEC message
  r420_msg_body_param_tlv_hdr_t rospec_start_trigger_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_ROSPEC_START_TRIGGER), // reserved=0, type=ROSpecStartTrigger
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(rospec_start_trigger_type))
  };

  uint8_t rospec_stop_trigger_type = 0; // Null - stop with STOP_ROSPEC message
  uint32_t rospec_duration_trigger_value = 0; // ignored in this case because stop trigger is Null
  r420_msg_body_param_tlv_hdr_t rospec_stop_trigger_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_ROSPEC_STOP_TRIGGER), // reserved=0, type=ROSpecStopTrigger
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(rospec_stop_trigger_type) + sizeof(rospec_duration_trigger_value))
  };

  r420_msg_body_param_tlv_hdr_t ro_boundary_spec_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_RO_BOUNDARY_SPEC), // reserved=0, type=ROBoundarySpec
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + ntohs(rospec_start_trigger_param_hdr.param_len) + ntohs(rospec_stop_trigger_param_hdr.param_len))
  };

  uint16_t antenna_count = 1;
  uint16_t antenna1_id = 1; // Antenna 1

  uint8_t aispec_stop_trigger_type = 0; // Null - stop when ROSpec is done
  uint32_t aispec_duration_trigger_value = 0; // ignored in this case because stop trigger is Null
    r420_msg_body_param_tlv_hdr_t aispec_stop_trigger_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_AISPEC_STOP_TRIGGER), // reserved=0, type=AISpecStopTrigger
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(aispec_stop_trigger_type) + sizeof(aispec_duration_trigger_value))
  };

  uint16_t hop_table_id = 1;
  uint16_t channel_index = 29; // 918.750 MHz
  uint16_t transmit_power = 81; // 30.00 dBm
  r420_msg_body_param_tlv_hdr_t rf_transmiter_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_RF_TRANSMITTER), // reserved=0, type=RFTransmitter
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(hop_table_id) + sizeof(channel_index) + sizeof(transmit_power))
  };

  uint8_t c1g2_tag_inventory_mask_flags = 0x40; // MB=1 ==> EPC memory bank
  uint16_t c1g2_tag_inventory_mask_bit_pointer = 0x20; // start at bit 32 (i.e., after the StoredCRC and StoredPC bits - that's where the EPC starts)
  //uint16_t c1g2_tag_inventory_mask_bit_count = 96; // length of EPC-96
  //uint8_t c1g2_tag_inventory_mask_value[12] = { 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb };
  //uint16_t c1g2_tag_inventory_mask_bit_count = 96;
  //uint8_t c1g2_tag_inventory_mask_value[12] = { 0x30, 0x2E, 0x33, 0x05, 0x48, 0xD0, 0xA8, 0x00, 0x00, 0x06, 0x08, 0xFD };
  //uint16_t c1g2_tag_inventory_mask_bit_count = 0;
  //uint8_t c1g2_tag_inventory_mask_value[0] = { };
  uint16_t c1g2_tag_inventory_mask_bit_count = 32;
  uint8_t c1g2_tag_inventory_mask_value[4] = { 0x14, 0xb0, 0x04, 0x73 };
  r420_msg_body_param_tlv_hdr_t c1g2_tag_inventory_mask_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_C1G2_TAG_INVENTORY_MASK), // reserved=0, type=C1G2TagInventoryMask
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(c1g2_tag_inventory_mask_flags) + sizeof(c1g2_tag_inventory_mask_bit_pointer) + sizeof(c1g2_tag_inventory_mask_bit_count) + sizeof(c1g2_tag_inventory_mask_value))
  };

  //uint8_t c1g2_filter_flags = 0x40; // T=1 ==> No truncate
  uint8_t c1g2_filter_flags = 0x00; // T=0 ==> Docs say the truncate flag must be always set to 0 (unspecified)
  r420_msg_body_param_tlv_hdr_t c1g2_filter_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_C1G2_FILTER), // reserved=0, type=C1G2Filter
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(c1g2_filter_flags) + ntohs(c1g2_tag_inventory_mask_param_hdr.param_len))
  };

  uint16_t mode_index = 1002; // AutoSet Static (default)
  int16_t tari = 0;
  r420_msg_body_param_tlv_hdr_t c1g2_rf_control_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_C1G2_RF_CONTROL), // reserved=0, type=C1G2RFControl
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(mode_index) + sizeof(tari))
  };

  uint8_t c1g2_singulation_control_flags = 0x40; // S=1 ==> Session 1
  uint16_t tag_population = 1; //default
  uint32_t tag_transit_time = 0; // default
  r420_msg_body_param_tlv_hdr_t c1g2_singulation_control_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_C1G2_SINGULATION_CONTROL), // reserved=0, type=C1G2SingulationControl
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(c1g2_singulation_control_flags) + sizeof(tag_population) + sizeof(tag_transit_time))
  };

  //InventorySearchMode
  uint32_t impinj_inventory_search_mode_vendor_id = IMPINJ_VENDOR_ID;
  uint32_t impinj_inventory_search_mode_subtype = R420_CUSTOM_PARAMETER_SUBTYPE_IMPINJ_INVENTORY_SEARCH_MODE;
  uint16_t inventory_search_mode = 2; //Dual Target Inventory
  r420_msg_body_param_tlv_hdr_t impinj_inventory_search_mode_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_CUSTOM_PARAMETER), // reserved=0, type=CustomParameter
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(impinj_inventory_search_mode_vendor_id) + sizeof(impinj_inventory_search_mode_subtype) + sizeof(inventory_search_mode))
  };

  uint8_t c1g2_inventory_command_flags = 0; // S=0 ==> so TagInventoryStateAware=0
  r420_msg_body_param_tlv_hdr_t c1g2_inventory_command_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_C1G2_INVENTORY_COMMAND), // reserved=0, type=C1G2InventoryCommand
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(c1g2_inventory_command_flags) + ntohs(c1g2_filter_param_hdr.param_len) + ntohs(c1g2_rf_control_param_hdr.param_len) + ntohs(c1g2_singulation_control_param_hdr.param_len) + ntohs(impinj_inventory_search_mode_param_hdr.param_len))
  };

  r420_msg_body_param_tlv_hdr_t antenna_configuration_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_ANTENNA_CONFIGURATION), // reserved=0, type=AntennaConfiguration
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(antenna1_id) + ntohs(rf_transmiter_param_hdr.param_len) + ntohs(c1g2_inventory_command_param_hdr.param_len))
  };

  uint16_t inventory_parameter_spec_id = 1;
  uint8_t protocol_id = 1; // EPCGlobalClass1Gen2
  r420_msg_body_param_tlv_hdr_t inventory_parameter_spec_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_INVENTORY_PARAMETER_SPEC), // reserved=0, type=InventoryParameterSpec
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(inventory_parameter_spec_id) + sizeof(protocol_id) + ntohs(antenna_configuration_param_hdr.param_len))
  };

  r420_msg_body_param_tlv_hdr_t aispec_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_AISPEC), // reserved=0, type=AISpec
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(antenna_count) + sizeof(antenna1_id) + ntohs(aispec_stop_trigger_param_hdr.param_len) + ntohs(inventory_parameter_spec_param_hdr.param_len))
  };

  r420_msg_body_param_tlv_hdr_t rospec_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_ROSPEC), // reserved=0, type=ROSpec
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(rospec_id) + sizeof(priority) + sizeof(current_state) + ntohs(ro_boundary_spec_param_hdr.param_len) + ntohs(aispec_param_hdr.param_len))
  };

  r420_msg_body_t body = { .buf = {0}, .len = ntohs(rospec_param_hdr.param_len) };

  /* Construct the body */
  size_t offset = 0;
  // Copy ROSpec tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = rospec_param_hdr;
  offset += sizeof(rospec_param_hdr);
  // Copy ROSpecID
  *(uint32_t *)(body.buf + offset) = htonl(rospec_id);
  offset += sizeof(rospec_id);
  // Copy Priority
  body.buf[offset] = priority;
  offset += sizeof(priority);
  // Copy CurrentState
  body.buf[offset] = current_state;
  offset += sizeof(current_state);
  // Copy ROBoundarySpec tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = ro_boundary_spec_param_hdr;
  offset += sizeof(ro_boundary_spec_param_hdr);
  // Copy ROSpecStartTrigger tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = rospec_start_trigger_param_hdr;
  offset += sizeof(rospec_start_trigger_param_hdr);
  // Copy ROSpecStartTriggerType
  body.buf[offset] = rospec_start_trigger_type;
  offset += sizeof(rospec_start_trigger_type);
  // Copy ROSpecStopTrigger tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = rospec_stop_trigger_param_hdr;
  offset += sizeof(rospec_stop_trigger_param_hdr);
  // Copy ROSpecStopTriggerType
  body.buf[offset] = rospec_stop_trigger_type;
  offset += sizeof(rospec_stop_trigger_type);
  // Copy ROSpecDurationTriggerValue
  *(uint32_t *)(body.buf + offset) = htonl(rospec_duration_trigger_value);
  offset += sizeof(rospec_duration_trigger_value);
  // Copy AISpec tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = aispec_param_hdr;
  offset += sizeof(aispec_param_hdr);
  // Copy AntennaCount
  *(uint16_t *)(body.buf + offset) = htons(antenna_count);
  offset += sizeof(antenna_count);
  // Copy AntennaID#1
  *(uint16_t *)(body.buf + offset) = htons(antenna1_id);
  offset += sizeof(antenna1_id);
  // Copy AISpecStopTrigger tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = aispec_stop_trigger_param_hdr;
  offset += sizeof(aispec_stop_trigger_param_hdr);
  // Copy AISpecStopTriggerType
  body.buf[offset] = aispec_stop_trigger_type;
  offset += sizeof(aispec_stop_trigger_type);
  // Copy AISpecStopTrigger's DurationTrigger
  *(uint32_t *)(body.buf + offset) = htonl(aispec_duration_trigger_value);
  offset += sizeof(aispec_duration_trigger_value);
  // Copy InventoryParameterSpec tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = inventory_parameter_spec_param_hdr;
  offset += sizeof(inventory_parameter_spec_param_hdr);
  // Copy InventoryParameterSpecID
  *(uint16_t *)(body.buf + offset) = htons(inventory_parameter_spec_id);
  offset += sizeof(inventory_parameter_spec_id);
  // Copy ProtocolID
  body.buf[offset] = protocol_id;
  offset += sizeof(protocol_id);
  // Copy AntennaConfiguration tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = antenna_configuration_param_hdr;
  offset += sizeof(antenna_configuration_param_hdr);
  // Copy AntennaID
  *(uint16_t *)(body.buf + offset) = htons(antenna1_id);
  offset += sizeof(antenna1_id);
  // Copy RFTransmitter tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = rf_transmiter_param_hdr;
  offset += sizeof(rf_transmiter_param_hdr);
  // Copy HopTableID
  *(uint16_t *)(body.buf + offset) = htons(hop_table_id);
  offset += sizeof(hop_table_id);
  // Copy ChannelIndex
  *(uint16_t *)(body.buf + offset) = htons(channel_index);
  offset += sizeof(channel_index);
  // Copy TransmitPower
  *(uint16_t *)(body.buf + offset) = htons(transmit_power);
  offset += sizeof(transmit_power);
  // Copy C1G2InventoryCommand tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = c1g2_inventory_command_param_hdr;
  offset += sizeof(c1g2_inventory_command_param_hdr);
  // Copy C1G2InventoryCommand flags
  body.buf[offset] = c1g2_inventory_command_flags;
  offset += sizeof(c1g2_inventory_command_flags);
  // Copy C1G2Filter tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = c1g2_filter_param_hdr;
  offset += sizeof(c1g2_filter_param_hdr);
  // Copy C1G2Filter flags
  body.buf[offset] = c1g2_filter_flags;
  offset += sizeof(c1g2_filter_flags);
  // Copy C1G2TagInventoryMask tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = c1g2_tag_inventory_mask_param_hdr;
  offset += sizeof(c1g2_tag_inventory_mask_param_hdr);
  // Copy C1G2TagInventoryMask flags
  body.buf[offset] = c1g2_tag_inventory_mask_flags;
  offset += sizeof(c1g2_tag_inventory_mask_flags);
  // Copy C1G2TagInventoryMask pointer
  *(uint16_t *)(body.buf + offset) = htons(c1g2_tag_inventory_mask_bit_pointer);
  offset += sizeof(c1g2_tag_inventory_mask_bit_pointer);
  // Copy C1G2TagInventoryMask MaskBitCount
  *(uint16_t *)(body.buf + offset) = htons(c1g2_tag_inventory_mask_bit_count);
  offset += sizeof(c1g2_tag_inventory_mask_bit_count);
  // Copy C1G2TagInventoryMask Tag Mask
  memcpy(body.buf + offset, c1g2_tag_inventory_mask_value, sizeof(c1g2_tag_inventory_mask_value));
  offset += sizeof(c1g2_tag_inventory_mask_value);
  // Copy C1G2RFControl tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = c1g2_rf_control_param_hdr;
  offset += sizeof(c1g2_rf_control_param_hdr);
  // Copy ModeIndex
  *(uint16_t *)(body.buf + offset) = htons(mode_index);
  offset += sizeof(mode_index);
  // Copy Tari
  *(int16_t *)(body.buf + offset) = htons(tari);
  offset += sizeof(tari);
  // Copy C1G2SingulationControl tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = c1g2_singulation_control_param_hdr;
  offset += sizeof(c1g2_singulation_control_param_hdr);
  // Copy C1G2SingulationControl flags
  body.buf[offset] = c1g2_singulation_control_flags;
  offset += sizeof(c1g2_singulation_control_flags);
  // Copy TagPopulation
  *(uint16_t *)(body.buf + offset) = htons(tag_population);
  offset += sizeof(tag_population);
  // Copy TagTransitTime
  *(uint32_t *)(body.buf + offset) = htonl(tag_transit_time);
  offset += sizeof(tag_transit_time);
  // Copy ImpinjInventorySearchMode tlv header
  *(r420_msg_body_param_tlv_hdr_t *)(body.buf + offset) = impinj_inventory_search_mode_param_hdr;
  offset += sizeof(impinj_inventory_search_mode_param_hdr);
  // Copy ImpinjInventorySearchMode vendor ID
  *(uint32_t *)(body.buf + offset) = htonl(impinj_inventory_search_mode_vendor_id);
  offset += sizeof(impinj_inventory_search_mode_vendor_id);
  // Copy ImpinjInventorySearchMode subtype
  *(uint32_t *)(body.buf + offset) = htonl(impinj_inventory_search_mode_subtype);
  offset += sizeof(impinj_inventory_search_mode_subtype);
  // Copy InventorySearchMode
  *(uint16_t *)(body.buf + offset) = htons(inventory_search_mode);
  offset += sizeof(inventory_search_mode);

  //body.len = offset; // Set final body length
  assert(offset == body.len);
  r420_msg_hdr_t hdr = {
    /* version = ctx->llrp_version, message type = R420_MSG_TYPE_ADD_ROSPEC */
    .attrs = htons((pCtx->llrp_version << 10) | R420_MSG_TYPE_ADD_ROSPEC),
    .message_length = htonl(sizeof(r420_msg_hdr_t) + body.len),
    .message_id = htonl(pCtx->next_tx_msg_id)
  };
  assert(R420_MSG_HDR_VERSION(hdr) == pCtx->llrp_version);
  assert(R420_MSG_HDR_MESSAGE_TYPE(hdr) == R420_MSG_TYPE_ADD_ROSPEC);
  assert(R420_MSG_HDR_MSG_LENGTH(hdr) == sizeof(r420_msg_hdr_t) + body.len);
  assert(R420_MSG_HDR_MSG_ID(hdr) == pCtx->next_tx_msg_id);
  r420_send_message(pCtx, &hdr, &body);
}

void r420_send_delete_rospec_msg(r420_ctx_t* pCtx) {
  pCtx->rospec_replace_flag = 1;

  uint32_t rospec_id = 1;

  r420_msg_body_t body = { .buf = {0}, .len = sizeof(rospec_id) };
  *(uint32_t *)(body.buf) = htonl(rospec_id);

  r420_msg_hdr_t hdr = {
    /* version = ctx->llrp_version, message type = R420_MSG_TYPE_DELETE_ROSPEC */
    .attrs = htons((pCtx->llrp_version << 10) | R420_MSG_TYPE_DELETE_ROSPEC),
    .message_length = htonl(sizeof(r420_msg_hdr_t) + body.len),
    .message_id = htonl(pCtx->next_tx_msg_id)
  };
  assert(R420_MSG_HDR_VERSION(hdr) == pCtx->llrp_version);
  assert(R420_MSG_HDR_MESSAGE_TYPE(hdr) == R420_MSG_TYPE_DELETE_ROSPEC);
  assert(R420_MSG_HDR_MSG_LENGTH(hdr) == sizeof(r420_msg_hdr_t) + body.len);
  assert(R420_MSG_HDR_MSG_ID(hdr) == pCtx->next_tx_msg_id);

  r420_send_message(pCtx, &hdr, &body);
}

void r420_send_enable_rospec_msg(r420_ctx_t* pCtx) {
  uint32_t rospec_id = 1;

  r420_msg_body_t body = { .buf = {0}, .len = sizeof(rospec_id) };
  *(uint32_t *)(body.buf) = htonl(rospec_id);

  r420_msg_hdr_t hdr = {
    /* version = ctx->llrp_version, message type = R420_MSG_TYPE_ENABLE_ROSPEC */
    .attrs = htons((pCtx->llrp_version << 10) | R420_MSG_TYPE_ENABLE_ROSPEC),
    .message_length = htonl(sizeof(r420_msg_hdr_t) + body.len),
    .message_id = htonl(pCtx->next_tx_msg_id)
  };
  assert(R420_MSG_HDR_VERSION(hdr) == pCtx->llrp_version);
  assert(R420_MSG_HDR_MESSAGE_TYPE(hdr) == R420_MSG_TYPE_ENABLE_ROSPEC);
  assert(R420_MSG_HDR_MSG_LENGTH(hdr) == sizeof(r420_msg_hdr_t) + body.len);
  assert(R420_MSG_HDR_MSG_ID(hdr) == pCtx->next_tx_msg_id);

  r420_send_message(pCtx, &hdr, &body);
}

void r420_send_disable_rospec_msg(r420_ctx_t* pCtx) {
  uint32_t rospec_id = 1;

  r420_msg_body_t body = { .buf = {0}, .len = sizeof(rospec_id) };
  *(uint32_t *)(body.buf) = htonl(rospec_id);

  r420_msg_hdr_t hdr = {
    /* version = ctx->llrp_version, message type = R420_MSG_TYPE_DISABLE_ROSPEC */
    .attrs = htons((pCtx->llrp_version << 10) | R420_MSG_TYPE_DISABLE_ROSPEC),
    .message_length = htonl(sizeof(r420_msg_hdr_t) + body.len),
    .message_id = htonl(pCtx->next_tx_msg_id)
  };
  assert(R420_MSG_HDR_VERSION(hdr) == pCtx->llrp_version);
  assert(R420_MSG_HDR_MESSAGE_TYPE(hdr) == R420_MSG_TYPE_DISABLE_ROSPEC);
  assert(R420_MSG_HDR_MSG_LENGTH(hdr) == sizeof(r420_msg_hdr_t) + body.len);
  assert(R420_MSG_HDR_MSG_ID(hdr) == pCtx->next_tx_msg_id);

  r420_send_message(pCtx, &hdr, &body);
}

void r420_send_start_rospec_msg(r420_ctx_t* pCtx) {
  uint32_t rospec_id = 1;

  r420_msg_body_t body = { .buf = {0}, .len = sizeof(rospec_id) };
  *(uint32_t *)(body.buf) = htonl(rospec_id);

  r420_msg_hdr_t hdr = {
    /* version = ctx->llrp_version, message type = R420_MSG_TYPE_START_ROSPEC */
    .attrs = htons((pCtx->llrp_version << 10) | R420_MSG_TYPE_START_ROSPEC),
    .message_length = htonl(sizeof(r420_msg_hdr_t) + body.len),
    .message_id = htonl(pCtx->next_tx_msg_id)
  };
  assert(R420_MSG_HDR_VERSION(hdr) == pCtx->llrp_version);
  assert(R420_MSG_HDR_MESSAGE_TYPE(hdr) == R420_MSG_TYPE_START_ROSPEC);
  assert(R420_MSG_HDR_MSG_LENGTH(hdr) == sizeof(r420_msg_hdr_t) + body.len);
  assert(R420_MSG_HDR_MSG_ID(hdr) == pCtx->next_tx_msg_id);

  r420_send_message(pCtx, &hdr, &body);
}

void r420_send_stop_rospec_msg(r420_ctx_t* pCtx) {
  uint32_t rospec_id = 1;

  r420_msg_body_t body = { .buf = {0}, .len = sizeof(rospec_id) };
  *(uint32_t *)(body.buf) = htonl(rospec_id);

  r420_msg_hdr_t hdr = {
    /* version = ctx->llrp_version, message type = R420_MSG_TYPE_STOP_ROSPEC */
    .attrs = htons((pCtx->llrp_version << 10) | R420_MSG_TYPE_STOP_ROSPEC),
    .message_length = htonl(sizeof(r420_msg_hdr_t) + body.len),
    .message_id = htonl(pCtx->next_tx_msg_id)
  };
  assert(R420_MSG_HDR_VERSION(hdr) == pCtx->llrp_version);
  assert(R420_MSG_HDR_MESSAGE_TYPE(hdr) == R420_MSG_TYPE_STOP_ROSPEC);
  assert(R420_MSG_HDR_MSG_LENGTH(hdr) == sizeof(r420_msg_hdr_t) + body.len);
  assert(R420_MSG_HDR_MSG_ID(hdr) == pCtx->next_tx_msg_id);

  r420_send_message(pCtx, &hdr, &body);
}

void r420_process_message(const r420_ctx_t *pCtx, const r420_msg_hdr_t *pHdr, const r420_msg_body_t *pBody) {
  r420_logf(pCtx, "Received message: Type=0x%X, Length=%u, ID=%u", R420_MSG_HDR_MESSAGE_TYPE(*pHdr), R420_MSG_HDR_MSG_LENGTH(*pHdr), R420_MSG_HDR_MSG_ID(*pHdr));
  switch(R420_MSG_HDR_MESSAGE_TYPE(*pHdr)) {
    case R420_MSG_TYPE_READER_EVENT_NOTIFICATION:
      r420_process_reader_event_notification_msg(pCtx, pBody);
      if (pCtx->rospec_started) {
        // For now, stop the ROSpec after receiving the first ReaderEventNotification
        r420_send_stop_rospec_msg((r420_ctx_t*)pCtx);
      } else {
        //r420_send_get_supported_version_msg((r420_ctx_t*)pCtx);
        r420_send_impinj_enable_extensions_msg((r420_ctx_t*)pCtx);
      }
      break;
    case R420_MSG_TYPE_GET_SUPPORTED_VERSION_RESPONSE:
      r420_process_get_supported_version_response_msg(pCtx, pBody);
      break;
    case R420_MSG_TYPE_GET_READER_CAPABILITIES_RESPONSE:
      r420_process_get_reader_capabilities_response_msg(pCtx, pBody);
      r420_send_set_reader_config_msg((r420_ctx_t*)pCtx);
      break;
    case R420_MSG_TYPE_CUSTOM_MESSAGE:
      assert(pBody->len >= 5);
      uint8_t subtype = *(pBody->buf + 4);
      switch (subtype) {
        case R420_CUSTOM_MESSAGE_SUBTYPE_IMPINJ_ENABLE_EXTENSIONS_RESPONSE:
          r420_process_impinj_enable_extensions_response_msg(pCtx, pBody);
          r420_send_get_reader_capabilities_msg((r420_ctx_t*)pCtx);
          break;
        default:
          r420_logf(pCtx, "Unhandled custom message subtype: 0x%X", subtype);
          r420_send_stop_rospec_msg((r420_ctx_t*)pCtx);
          break;
      }
      break;
    case R420_MSG_TYPE_GET_READER_CONFIG_RESPONSE:
      r420_process_get_reader_config_response_msg(pCtx, pBody);
      r420_send_get_rospecs_msg((r420_ctx_t*)pCtx);
      break;
    case R420_MSG_TYPE_SET_READER_CONFIG_RESPONSE:
      r420_process_set_reader_config_response_msg((r420_ctx_t*)pCtx, pBody);
      r420_send_get_reader_config_msg((r420_ctx_t*)pCtx);
      break;
    case R420_MSG_TYPE_GET_ROSPECS_RESPONSE:
      r420_process_get_rospecs_response_msg((r420_ctx_t*)pCtx, pBody);
      if (pCtx->rospec_added) {
        if (REPLACE_ROSPEC == 1 && !pCtx->rospec_replace_flag) {
          r420_send_delete_rospec_msg((r420_ctx_t*)pCtx);
        } else if (pCtx->rospec_enabled) {
          if (pCtx->rospec_started) {
            r420_logf(pCtx, "ROSPEC already started. No further action.");
          } else {
            r420_send_start_rospec_msg((r420_ctx_t*)pCtx);
            //r420_send_disable_rospec_msg((r420_ctx_t*)pCtx);
            // r420_unloop((r420_ctx_t*)pCtx); // Terminate
          }
        } else {
          r420_send_enable_rospec_msg((r420_ctx_t*)pCtx);
        }
      } else {
        r420_send_add_rospec_msg((r420_ctx_t*)pCtx);
      }
      break;
    case R420_MSG_TYPE_ADD_ROSPEC_RESPONSE:
      r420_process_add_rospec_response_msg((r420_ctx_t*)pCtx, pBody);
      r420_send_get_rospecs_msg((r420_ctx_t*)pCtx);
      break;
    case R420_MSG_TYPE_DELETE_ROSPEC_RESPONSE:
      r420_process_delete_rospec_response_msg((r420_ctx_t*)pCtx, pBody);
      r420_send_get_rospecs_msg((r420_ctx_t*)pCtx);
      break;
    case R420_MSG_TYPE_ENABLE_ROSPEC_RESPONSE:
      r420_process_enable_rospec_response_msg((r420_ctx_t*)pCtx, pBody);
      r420_send_get_rospecs_msg((r420_ctx_t*)pCtx);
      break;
    case R420_MSG_TYPE_DISABLE_ROSPEC_RESPONSE:
      r420_process_disable_rospec_response_msg((r420_ctx_t*)pCtx, pBody);
      r420_send_get_rospecs_msg((r420_ctx_t*)pCtx);
      break;
    case R420_MSG_TYPE_START_ROSPEC_RESPONSE:
      r420_process_start_rospec_response_msg((r420_ctx_t*)pCtx, pBody);
      r420_send_get_rospecs_msg((r420_ctx_t*)pCtx);
      break;
    case R420_MSG_TYPE_STOP_ROSPEC_RESPONSE:
      r420_process_stop_rospec_response_msg((r420_ctx_t*)pCtx, pBody);
      //r420_send_get_rospecs_msg((r420_ctx_t*)pCtx);
      r420_send_disable_rospec_msg((r420_ctx_t*)pCtx);
      r420_unloop((r420_ctx_t*)pCtx); // Terminate
      break;
    case R420_MSG_TYPE_RO_ACCESS_REPORT:
      r420_process_ro_access_report_msg(pCtx, pBody);
      break;
    default:
      //assert(0);
      r420_logf(pCtx, "Unhandled message type: 0x%X", R420_MSG_HDR_MESSAGE_TYPE(*pHdr));
      r420_send_stop_rospec_msg((r420_ctx_t*)pCtx);
      break;
  }


  // if (pBody->len > 0))
}

void r420_loop(r420_ctx_t *pCtx, void* pArg) {
  assert(pCtx->fd >= 0);
  pCtx->terminate_flag = 0; // Reset terminate flag
  pCtx->pUserData = pArg;

  while (!pCtx->terminate_flag) {
    r420_logf(pCtx, "Waiting for message...");
    r420_msg_hdr_t hdr = r420_receive_header(pCtx);
    r420_msg_body_t body = r420_receive_body(pCtx, R420_MSG_HDR_MSG_LENGTH(hdr) - sizeof(hdr));
    r420_process_message(pCtx, &hdr, &body);

    if (pCtx->loop_handler) {
      pCtx->loop_handler(pCtx);
    }
  }
}

void r420_unloop(r420_ctx_t *pCtx) {
  pCtx->terminate_flag = 1;
}

void r420_stop(r420_ctx_t *pCtx) {
  r420_send_stop_rospec_msg(pCtx);
}