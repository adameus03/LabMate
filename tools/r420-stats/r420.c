#include <sys/socket.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include "r420.h"

#define R420_MSG_BODY_MAX_SIZE 4096

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
#define R420_MSG_BODY_PARAM_TV_HDR_TYPE(hdr) (ntohs((hdr).attrs) & 0x7F)

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
    char buf[256];
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
  return (r420_ctx_t){ .fd = fd, .loop_handler = NULL, .log_handler = NULL, .next_tx_msg_id = 1, .llrp_version = 1, .terminate_flag = 0, .rospec_added = 0 };
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
  uint16_t type;
  uint16_t len;
  size_t value_offset; // Offset of the value field within the parameter (after header)
} r420_msg_param_info_t;

r420_msg_param_info_t r420_process_param(const r420_msg_body_t *pBody, size_t offset) {
  assert(offset < pBody->len);
  if (pBody->buf[0] & 0x80) { // check if the parameter is TLV or TV encoded
    // TV encoded
    r420_msg_body_param_tv_hdr_t tv_hdr = *(r420_msg_body_param_tv_hdr_t *)(pBody->buf + offset);
    uint8_t type = R420_MSG_BODY_PARAM_TV_HDR_TYPE(tv_hdr);
    uint16_t len = 0xffff; // TV has no length field
    switch (type) {
      // TODO We should handle all TV parameter lengths here
      default:
        break;
    }
    return (r420_msg_param_info_t){ .type = type, .len = 0xffff, .value_offset = offset + sizeof(tv_hdr) }; // TV has no length field
  } else {
    // TLV encoded
    r420_msg_body_param_tlv_hdr_t tlv_hdr = *(r420_msg_body_param_tlv_hdr_t *)(pBody->buf + offset);
    uint16_t type = R420_MSG_BODY_PARAM_TLV_HDR_TYPE(tlv_hdr);
    uint16_t len = R420_MSG_BODY_PARAM_TLV_HDR_LEN(tlv_hdr);
    return (r420_msg_param_info_t){ .type = type, .len = len, .value_offset = offset + sizeof(tlv_hdr) };
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
    r420_msg_param_info_t param_info = r420_process_param(pBody, offset);
    switch(param_info.type) {
      case R420_PARAM_TYPE_UTC_TIMESTAMP:
        break;
      case R420_PARAM_TYPE_CONNECTION_ATTEMPT_EVENT:
        uint16_t status = *(uint16_t *)(pBody->buf + param_info.value_offset);
        assert(status == 0); // We expect status=Success
        break;
      default:
        // Unknown parameter type, we just skip it
        //TODO Add a ctx handler to pass this event to the user
        break;
    }
    assert(param_info.len != 0xffff); // Make sure there are no unsupported TV params as that would break the parsing
    offset = param_info.value_offset + param_info.len;
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
  r420_msg_param_info_t param_info = r420_process_param(pBody, 2);
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
  r420_msg_param_info_t param_info = r420_process_param(pBody, 0);
  assert(param_info.type == R420_PARAM_TYPE_LLRP_STATUS);
  assert(param_info.len >= 4); // We expect at least 4 bytes for LLRPStatus
  uint16_t status_code = *(uint16_t *)(pBody->buf + param_info.value_offset);
  assert(status_code == 0); // We expect StatusCode=M_Success
  //uint16_t err_desc_len = *(uint16_t *)(pBody->buf + param_info.value_offset + 2);
  // We ignore the error description and error params for now
  // TODO Handle the error description and error params
  // TODO Parse other parameters in the GetReaderCapabilitiesResponse message
}

void r420_process_get_reader_config_response_msg(const r420_ctx_t *pCtx, const r420_msg_body_t *pBody) {
  // First 4 bytes - TLV header for LLRPStatus parameter
  // Next 2 bytes - StatusCode
  // Next 2 bytes - Error Description ByteCount
  // ...
  assert(pBody->len >= 8); // We expect at least 8 bytes
  r420_msg_param_info_t param_info = r420_process_param(pBody, 0);
  assert(param_info.type == R420_PARAM_TYPE_LLRP_STATUS);
  assert(param_info.len >= 4); // We expect at least 4 bytes for LLRPStatus
  uint16_t status_code = *(uint16_t *)(pBody->buf + param_info.value_offset);
  assert(status_code == 0); // We expect StatusCode=M
  //uint16_t err_desc_len = *(uint16_t *)(pBody->buf + param_info.value_offset + 2);
  // We ignore the error description and error params for now
  // TODO Handle the error description and error params
  // TODO Parse other parameters in the GetReaderConfigResponse message
}

void r420_process_get_rospecs_response_msg(const r420_ctx_t *pCtx, const r420_msg_body_t *pBody) {
  // First 4 bytes - TLV header for LLRPStatus parameter
  // Next 2 bytes - StatusCode
  // Next 2 bytes - Error Description ByteCount
  // ...
  assert(pBody->len >= 8); // We expect at least 8 bytes
  r420_msg_param_info_t param_info = r420_process_param(pBody, 0);
  assert(param_info.type == R420_PARAM_TYPE_LLRP_STATUS);
  assert(param_info.len >= 4); // We expect at least 4 bytes for LLRPStatus
  uint16_t status_code = *(uint16_t *)(pBody->buf + param_info.value_offset);
  assert(status_code == 0); // We expect StatusCode=M_Success
  //uint16_t err_desc_len = *(uint16_t *)(pBody->buf + param_info.value_offset + 2);
  // We ignore the error description and error params for now
  // TODO Handle the error description and error params
  // TODO Parse other parameters in the GetROSpecsResponse message
}

void r420_process_add_rospec_response_msg(r420_ctx_t *pCtx, const r420_msg_body_t *pBody) {
  // First 4 bytes - TLV header for LLRPStatus parameter
  // Next 2 bytes - StatusCode
  // Next 2 bytes - Error Description ByteCount
  // ...
  assert(pBody->len >= 8); // We expect at least 8 bytes
  r420_msg_param_info_t param_info = r420_process_param(pBody, 0);
  assert(param_info.type == R420_PARAM_TYPE_LLRP_STATUS);
  assert(param_info.len >= 4); // We expect at least 4 bytes for LLRPStatus
  uint16_t status_code = *(uint16_t *)(pBody->buf + param_info.value_offset);
  assert(status_code == 0); // We expect StatusCode=M_Success
  pCtx->rospec_added = 1;
  //uint16_t err_desc_len = *(uint16_t *)(pBody->buf + param_info.value_offset + 2);
  // We ignore the error description and error params for now
  // TODO Handle the error description and error params
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
  r420_msg_body_t body = { .buf = {0}, .len = sizeof(requested_data) };
  body.buf[0] = requested_data;

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

void r420_send_get_reader_config_msg(r420_ctx_t* pCtx) {
  uint16_t antenna_id = 0; // 0 means all antennas
  uint8_t requested_data = 0; // 0 means all config data
  uint16_t gpi_port_num = 0; // 0 means all GPI ports
  uint16_t gpo_port_num = 0; // 0 means all GPO ports

  r420_msg_body_t body = { .buf = {0}, .len = sizeof(antenna_id) + sizeof(requested_data) + sizeof(gpi_port_num) + sizeof(gpo_port_num) };
  *(uint16_t *)(body.buf) = htons(antenna_id);
  body.buf[2] = requested_data;
  *(uint16_t *)(body.buf + 3) = htons(gpi_port_num);
  *(uint16_t *)(body.buf + 5) = htons(gpo_port_num);

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

  uint16_t inventory_parameter_spec_id = 1;
  uint8_t protocol_id = 1; // EPCGlobalClass1Gen2
  r420_msg_body_param_tlv_hdr_t inventory_parameter_spec_param_hdr = {
    .attrs = htons((0 << 10) | R420_PARAM_TYPE_INVENTORY_PARAMETER_SPEC), // reserved=0, type=InventoryParameterSpec
    .param_len = htons(sizeof(r420_msg_body_param_tlv_hdr_t) + sizeof(inventory_parameter_spec_id) + sizeof(protocol_id))
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
  body.len = offset; // Set final body length
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

void r420_process_message(const r420_ctx_t *pCtx, const r420_msg_hdr_t *pHdr, const r420_msg_body_t *pBody) {
  r420_logf(pCtx, "Received message: Type=0x%X, Length=%u, ID=%u", R420_MSG_HDR_MESSAGE_TYPE(*pHdr), R420_MSG_HDR_MSG_LENGTH(*pHdr), R420_MSG_HDR_MSG_ID(*pHdr));
  switch(R420_MSG_HDR_MESSAGE_TYPE(*pHdr)) {
    case R420_MSG_TYPE_READER_EVENT_NOTIFICATION:
      r420_process_reader_event_notification_msg(pCtx, pBody);
      //r420_send_get_supported_version_msg((r420_ctx_t*)pCtx);
      r420_send_get_reader_capabilities_msg((r420_ctx_t*)pCtx);
      break;
    case R420_MSG_TYPE_GET_SUPPORTED_VERSION_RESPONSE:
      r420_process_get_supported_version_response_msg(pCtx, pBody);
      break;
    case R420_MSG_TYPE_GET_READER_CAPABILITIES_RESPONSE:
      r420_process_get_reader_capabilities_response_msg(pCtx, pBody);
      r420_send_get_reader_config_msg((r420_ctx_t*)pCtx);
      break;
    case R420_MSG_TYPE_GET_READER_CONFIG_RESPONSE:
      r420_process_get_reader_config_response_msg(pCtx, pBody);
      r420_send_get_rospecs_msg((r420_ctx_t*)pCtx);
      break;
    case R420_MSG_TYPE_GET_ROSPECS_RESPONSE:
      r420_process_get_rospecs_response_msg(pCtx, pBody);
      if (!pCtx->rospec_added) {
        r420_send_add_rospec_msg((r420_ctx_t*)pCtx);
      }
      break;
    case R420_MSG_TYPE_ADD_ROSPEC_RESPONSE:
      r420_process_add_rospec_response_msg((r420_ctx_t*)pCtx, pBody);
      r420_send_get_rospecs_msg((r420_ctx_t*)pCtx);
      break;
    default:
      assert(0);
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
