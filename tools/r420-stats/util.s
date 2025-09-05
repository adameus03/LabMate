.section .data

string_newline:
  .ascii "\n"
string_newline_len = . - string_newline
string_space:
  .ascii " "
string_space_len = . - string_space
string_open_paren:
  .ascii "("
string_open_paren_len = . - string_open_paren
string_close_paren:
  .ascii ")"
string_close_paren_len = . - string_close_paren
string_0x:
  .ascii "0x"
string_0x_len = . - string_0x

hexbuf:
  .space 2

.section .text
.globl print_al_hex
.globl print_ah_hex
.globl print_bl_hex
.globl print_bh_hex
.globl print_newline
.globl print_space
.globl print_open_paren
.globl print_close_paren
.globl print_0x

print_0x:
  movq $1, %rax
  movq $1, %rdi
  leaq string_0x(%rip), %rsi
  movq $string_0x_len, %rdx
  syscall
  ret

print_al_hex:
  push %rax
  push %rbx
  push %rdx
  movb %al, %bl

  # high nibble
  movb %bl, %al
  shrb $4, %al
  call nibble_to_ascii
  movb %al, hexbuf

  # low nibble
  movb %bl, %al
  andb $0xF, %al
  call nibble_to_ascii
  movb %al, hexbuf+1

  # write(1, hexbuf, 2)
  movq $1, %rax
  movq $1, %rdi
  leaq hexbuf(%rip), %rsi
  movq $2, %rdx
  syscall

  pop %rdx
  pop %rbx
  pop %rax
  ret

print_ah_hex:
  push %rax
  movb %ah, %al
  call print_al_hex
  pop %rax
  ret

print_bl_hex:
  push %rax
  movb %bl, %al
  call print_al_hex
  pop %rax
  ret

print_bh_hex:
  push %rax
  movb %bh, %al
  call print_al_hex
  pop %rax
  ret

nibble_to_ascii:
  cmpb $9, %al
  jg nibble_to_ascii_letter
  addb $'0', %al
  ret
  nibble_to_ascii_letter:
    addb $('A'-10), %al
    ret

print_newline:
  movq $1, %rax
  movq $1, %rdi
  leaq string_newline(%rip), %rsi
  movq $string_newline_len, %rdx
  syscall
  ret

print_space:
  movq $1, %rax
  movq $1, %rdi
  leaq string_space(%rip), %rsi
  movq $string_space_len, %rdx
  syscall
  ret

print_open_paren:
  movq $1, %rax
  movq $1, %rdi
  leaq string_open_paren(%rip), %rsi
  movq $string_open_paren_len, %rdx
  syscall
  ret

print_close_paren:
  movq $1, %rax
  movq $1, %rdi
  leaq string_close_paren(%rip), %rsi
  movq $string_close_paren_len, %rdx
  syscall
  ret

