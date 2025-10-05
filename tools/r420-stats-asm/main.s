.section .data

string_msg_version_label:
  .ascii "Version: "
string_msg_version_label_len = . - string_msg_version_label
string_msg_version_llrp_v_1_0_1:
  .ascii "LLRP 1.0.1"
string_msg_version_llrp_v_1_0_1_len = . - string_msg_version_llrp_v_1_0_1
string_msg_version_llrp_v_1_1:
  .ascii "LLRP 1.1"
string_msg_version_llrp_v_1_1_len = . - string_msg_version_llrp_v_1_1
string_msg_version_llrp_v_2_0:
  .ascii "LLRP 2.0"
string_msg_version_llrp_v_2_0_len = . - string_msg_version_llrp_v_2_0

string_msg_type_label:
  .ascii "Message Type: "
string_msg_type_label_len = . - string_msg_type_label

string_msg_length_label:
  .ascii "Message Length: "
string_msg_length_label_len = . - string_msg_length_label

string_msg_id_label:
  .ascii "Message ID: "
string_msg_id_label_len = . - string_msg_id_label

server_addr:
  .short 2 # AF_INET = 2
  .short 0xdc13 # port 5084 (0x13dc big-endian stored as 0xdc13)
  # .short 0xa00f # port 4000 (0x0fa0 big-endian stored as 0xa00f)
  .long 0x2b12a8c0 # IP 192.168.18.43 (0xc0a8122b big-endian stored as 0x2b12a8c0)
  # .long 0x0100007f # IP 127.0.0.1 (0x7f000001 big-endian stored as 0x0100007f)
  .space 8, 0 # sin_zero[8]

error_msg_header_rsvd_nonzero = 1
error_msg_header_ver_unknown = 2
error_receive_msg_header_overflow = 3 # should never happen
error_connection_closed_by_peer = 4
error_receive_msg_body_read_overflow = 5 # should never happen
error_not_implemented = 6
error_msg_body_param_tlv_rsvd_nonzero = 7
error_msg_body_length_exceeds_buffer = 8

msg_header_buf_size = 10
msg_header_buf:
  .space msg_header_buf_size, 0 # buffer for reading message header

msg_body_buf_size = 256
msg_body_buf:
  .space msg_body_buf_size, 0 # buffer for reading message body

.section .text
.global _start

# Check if msg_header_buf first byte has 3 most significant bits zero
assert_msg_header_rsvd_zero:
  movb msg_header_buf(%rip), %al
  andb $0xe0, %al # mask with 11100000
  cmpb $0, %al
  jne assert_msg_header_rsvd_zero_fail
  ret
  assert_msg_header_rsvd_zero_fail:
    movq $error_msg_header_rsvd_nonzero, %rdi
    movq $60, %rax # syscall: exit
    syscall

print_msg_ver:
  movb msg_header_buf(%rip), %al
  andb $0x1c, %al # mask with 00011100
  shrb $2, %al # shift right by 2 to get version in least significant bits
  call print_al_hex
  ret

print_msg_type:
  movb msg_header_buf(%rip), %ah
  movb msg_header_buf+1(%rip), %al
  andw $0x3ff, %ax # mask with 00000011 11111111
  # ax now has the message type
  call print_ah_hex
  call print_al_hex
  ret

print_msg_len:
  movb msg_header_buf+2(%rip), %ah
  movb msg_header_buf+3(%rip), %al
  movb msg_header_buf+4(%rip), %bh
  movb msg_header_buf+5(%rip), %bl
  # ax:bx now has the message length
  call print_ah_hex
  call print_al_hex
  call print_bh_hex
  call print_bl_hex
  ret

print_msg_id:
  movb msg_header_buf+6(%rip), %ah
  movb msg_header_buf+7(%rip), %al
  movb msg_header_buf+8(%rip), %bh
  movb msg_header_buf+9(%rip), %bl
  # ax:bx now has the message id
  call print_ah_hex
  call print_al_hex
  call print_bh_hex
  call print_bl_hex
  ret

print_msg_ver_description:
  movb msg_header_buf(%rip), %al
  andb $0x1c, %al # mask with 00011100
  shrb $2, %al # shift right by 2 to get version in least significant bits
  cmpb $1, %al
  je print_msg_ver_description_llrp_v_1_0_1
  cmpb $2, %al
  je print_msg_ver_description_llrp_v_1_1
  cmpb $3, %al
  je print_msg_ver_description_llrp_v_2_0
  # unknown version
  movq $error_msg_header_ver_unknown, %rdi
  movq $60, %rax # syscall: exit
  syscall

  print_msg_ver_description_llrp_v_1_0_1:
    movq $1, %rax            # syscall: write
    movq $1, %rdi            # fd = stdout
    leaq string_msg_version_llrp_v_1_0_1(%rip), %rsi
    movq $string_msg_version_llrp_v_1_0_1_len, %rdx
    syscall
    ret

  print_msg_ver_description_llrp_v_1_1:
    movq $1, %rax            # syscall: write
    movq $1, %rdi            # fd = stdout
    leaq string_msg_version_llrp_v_1_1(%rip), %rsi
    movq $string_msg_version_llrp_v_1_1_len, %rdx
    syscall
    ret

  print_msg_ver_description_llrp_v_2_0:
    movq $1, %rax            # syscall: write
    movq $1, %rdi            # fd = stdout
    leaq string_msg_version_llrp_v_2_0(%rip), %rsi
    movq $string_msg_version_llrp_v_2_0_len, %rdx
    syscall
    ret

print_msg_ver_wrapped:
  # print "Version: "
  movq $1, %rax            # syscall: write
  movq $1, %rdi            # fd = stdout
  leaq string_msg_version_label(%rip), %rsi
  movq $string_msg_version_label_len, %rdx
  syscall

  call print_0x
  call print_msg_ver
  call print_space
  call print_open_paren
  call print_msg_ver_description
  call print_close_paren
  call print_newline
  ret

print_msg_type_wrapped:
  # print "Message Type: "
  movq $1, %rax            # syscall: write
  movq $1, %rdi            # fd = stdout
  leaq string_msg_type_label(%rip), %rsi
  movq $string_msg_type_label_len, %rdx
  syscall

  call print_0x
  call print_msg_type
  call print_newline
  ret

print_msg_length_wrapped:
  # print "Message Length: "
  movq $1, %rax            # syscall: write
  movq $1, %rdi            # fd = stdout
  leaq string_msg_length_label(%rip), %rsi
  movq $string_msg_length_label_len, %rdx
  syscall

  call print_0x
  call print_msg_len
  call print_newline
  ret

print_msg_id_wrapped:
  # print "Message ID: "
  movq $1, %rax            # syscall: write
  movq $1, %rdi            # fd = stdout
  leaq string_msg_id_label(%rip), %rsi
  movq $string_msg_id_label_len, %rdx
  syscall

  call print_0x
  call print_msg_id
  call print_newline
  ret

# Assumes r12 contains the socket fd
# Assumes r13 contains the number of bytes read into msg_header_buf (should be initialized to 0 by the caller)
receive_msg_header:
  movq $0, %rax # syscall: read
  movq %r12, %rdi # fd
  leaq msg_header_buf(%rip), %rsi # msg_header_buf
  addq %r13, %rsi # msg_header_buf + num_bytes_read
  movq $msg_header_buf_size, %rdx # count
  subq %r13, %rdx # count = msg_header_buf_size - num_bytes_read
  syscall

  cmpq $0, %rax
  je receive_msg_header_connection_closed_by_peer
  
  addq %rax, %r13 # update num_bytes_read
  cmpq $msg_header_buf_size, %r13
  jl receive_msg_header
  jg receive_msg_header_overflow_error
  ret
  receive_msg_header_overflow_error:
    movq $error_receive_msg_header_overflow, %rdi
    movq $60, %rax # syscall: exit
    syscall
  receive_msg_header_connection_closed_by_peer:
    movq $error_connection_closed_by_peer, %rdi
    movq $60, %rax # syscall: exit
    syscall

# output the message length in eax
get_msg_length_from_header:
  movl msg_header_buf+2(%rip), %eax
  bswapl %eax
  # eax now has the message length
  ret

# assumes the first 3 bytes of msg_body_buf contain the TLV header
# outputs the parameter type into ax
get_tlv_param_type_from_body:
  movb msg_body_buf(%rip), %ah
  movb msg_body_buf+1(%rip), %al
  andw $0x3ff, %ax # mask with 00000011 11111111
  # ax now has the parameter type
  ret

# assumes the first 3 bytes of msg_body_buf contain the TLV header
# outputs the parameter length into ax
get_tlv_param_length_from_body:
  movb msg_body_buf+2(%rip), %ah
  movb msg_body_buf+3(%rip), %al
  # ax now has the parameter length
  ret

# Assumes r12 contains the socket fd
receive_msg_body:
  call get_msg_length_from_header
  cmpl $msg_header_buf_size, %eax
  jg receive_msg_body_nonempty
  ret
  receive_msg_body_nonempty:
    # rax now has the total message length
    # check if our body buffer is large enough
    subl $msg_header_buf_size, %eax
    cmpl $msg_body_buf_size, %eax
    jg receive_msg_body_length_exceeds_buffer_error

    # save message body length in r14d
    movl %eax, %r14d 
    # TODO

    # receive first byte of body
    movq $0, %rax # syscall: read
    movq %r12, %rdi # fd
    leaq msg_body_buf(%rip), %rsi # msg_body_buf
    movq $1, %rdx # count
    syscall

    # check for errors
    cmpq $0, %rax
    je receive_msg_body_connection_closed_by_peer
    cmpq $1, %rax
    jne receive_msg_body_read_overflow_error

    # check if the parameter is TLV or TV encoded
    movb msg_body_buf(%rip), %al
    andb $0x80, %al # mask with 10000000
    cmpb $0, %al
    je receive_msg_body_param_tlv

    # TV is not implemented yet
    movq $error_not_implemented, %rdi
    movq $60, %rax # syscall: exit
    syscall

  receive_msg_body_param_tlv:
    # check if the reserved bits are zero
    andb $0xfc, %al # mask with 11111100
    cmpb $0, %al
    jne receive_msg_body_param_tlv_rsvd_nonzero_error

    # Receive the next 3 bytes to complete the TLV header
    # Using r13 to count received bytes
    xorq %r13, %r13 # num bytes read = 0
    call receive_msg_body_param_tlv_complete_receiving_header

    # Now we have the complete TLV header in msg_body_buf and we are ready to get the param length from it
    call get_tlv_param_length_from_body
    # TODO


  
  # Assumes r13 is initialized to 0
  receive_msg_body_param_tlv_complete_receiving_header:
    movq $0, %rax # syscall: read
    movq %r12, %rdi # fd
    leaq msg_body_buf+1(%rip), %rsi # msg_body_buf + 1
    addq %r13, %rsi # msg_body_buf + 1 + num_bytes_read
    movq $3, %rdx # count
    subq %r13, %rdx # count = 3 - num_bytes_read
    syscall

    cmpq $0, %rax
    je receive_msg_body_connection_closed_by_peer

    addq %rax, %r13 # update num_bytes_read
    cmpq $3, %rax
    jl receive_msg_body_param_tlv_complete_receiving_header
    jg receive_msg_body_read_overflow_error
    ret

  receive_msg_body_param_tlv_get_type:
    movb msg_body_buf(%rip), %ah
    movb msg_body_buf+1(%rip), %al
    andw $0x3ff, %ax # mask with 00000011 11111111
    # ax now has the parameter type
    ret

  receive_msg_body_param_tlv_get_len:
    movb msg_body_buf+2(%rip), %ah
    movb msg_body_buf+3(%rip), %al
    # ax now has the parameter length
    ret

  receive_msg_body_connection_closed_by_peer:
    movq $error_connection_closed_by_peer, %rdi
    movq $60, %rax # syscall: exit
    syscall
  receive_msg_body_read_overflow_error:
    movq $error_receive_msg_body_read_overflow, %rdi
    movq $60, %rax # syscall: exit
    syscall
  receive_msg_body_length_exceeds_buffer_error:
    movq $error_msg_body_length_exceeds_buffer, %rdi
    movq $60, %rax # syscall: exit
    syscall
  receive_msg_body_param_tlv_rsvd_nonzero_error:
    movq $error_msg_body_param_tlv_rsvd_nonzero, %rdi
    movq $60, %rax # syscall: exit

_start:
  # socket(AF_INET=2, SOCK_STREAM=1, 0)
  movq $41, %rax # syscall: socket
  movq $2, %rdi # AF_INET
  movq $1, %rsi # SOCK_STREAM
  xorq %rdx, %rdx # protocol 0
  syscall
  movq %rax, %r12 # save fd in r12

  # connect(fd, &server_addr, 16)
  movq $42, %rax # syscall: connect
  movq %r12, %rdi # fd
  leaq server_addr(%rip), %rsi # &server_addr
  movq $16, %rdx # addrlen
  syscall

  # read(fd, buf, msg_header_buf_size)
  # movq $0, %rax # syscall: read
  # movq %r12, %rdi # fd
  # leaq msg_header_buf(%rip), %rsi # msg_header_buf
  # movq $msg_header_buf_size, %rdx # count
  # syscall
  # movq %rax, %r13 # save num bytes read in r13

  xorq %r13, %r13 # num bytes read = 0
  call receive_msg_header

  call assert_msg_header_rsvd_zero
  call print_msg_ver_wrapped
  call print_msg_type_wrapped
  call print_msg_length_wrapped
  call print_msg_id_wrapped

  call receive_msg_body

  # write(1, buf, num_bytes_read)
  # movq $1, %rax # syscall: write
  # movq $1, %rdi # fd = stdout
  # leaq msg_header_buf(%rip), %rsi # msg_header_buf
  # movq %r13, %rdx # count = num bytes read
  # syscall

  # close(fd)
  movq $3, %rax # syscall: close
  movq %r12, %rdi # fd
  syscall

  # exit(0)
  movq $60, %rax # syscall: exit
  xorq %rdi, %rdi # status 0
  syscall

