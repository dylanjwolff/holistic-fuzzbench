/*
 * PRINT instrumentation.
 */

#include "stdlib.c"

static uint16_t SF = 0b1000000000000000;
static uint16_t ZF = 0b0100000000000000;
static uint16_t PF = 0b0000010000000000;
static uint16_t CF = 0b0000000100000000;
static uint16_t OF = 0b0000000000000001;

void view(off_t offset, const char *asm_instr, const size_t len)
{
    printf("\n%p: saw %s -- NONCE=X2M3BF73\n", offset, asm_instr);
}

void view_js_jns(off_t offset, const char *asm_instr, uint16_t rflags)
{
    if (!(rflags & SF)) {
            printf("\n%p: saw %s TRUE -- NONCE=X2M3BF73\n", offset, asm_instr);
    } else {
            printf("\n%p: saw %s FALSE -- NONCE=X2M3BF73\n", offset, asm_instr);
    }
}

void view_je_jne(off_t offset, const char *asm_instr, uint16_t rflags)
{
    if (!(rflags & ZF)) {
            printf("\n%p: saw %s TRUE -- NONCE=X2M3BF73\n", offset, asm_instr);
    } else {
            printf("\n%p: saw %s FALSE -- NONCE=X2M3BF73\n", offset, asm_instr);
    }
}

void view_jb_jnb(off_t offset, const char *asm_instr, uint16_t rflags)
{
    if (!(rflags & CF)) {
            printf("\n%p: saw %s TRUE -- NONCE=X2M3BF73\n", offset, asm_instr);
    } else {
            printf("\n%p: saw %s FALSE -- NONCE=X2M3BF73\n", offset, asm_instr);
    }
}

void view_jbe_ja(off_t offset, const char *asm_instr, uint16_t rflags)
{
    if (!!(rflags & CF) || !!(rflags & ZF)) {
            printf("\n%p: saw %s TRUE -- NONCE=X2M3BF73\n", offset, asm_instr);
    } else {
            printf("\n%p: saw %s FALSE -- NONCE=X2M3BF73\n", offset, asm_instr);
    }
}

void view_jl_jge(off_t offset, const char *asm_instr, uint16_t rflags)
{
    if (!(rflags & CF) != !(rflags & ZF)) {
            printf("\n%p: saw %s TRUE -- NONCE=X2M3BF73\n", offset, asm_instr);
    } else {
            printf("\n%p: saw %s FALSE -- NONCE=X2M3BF73\n", offset, asm_instr);
    }
}

void view_jle_jg(off_t offset, const char *asm_instr, uint16_t rflags)
{
    if (!(rflags & ZF) && (!(rflags & SF) == !(rflags & OF))) {
            printf("\n%p: saw %s TRUE -- NONCE=X2M3BF73\n", offset, asm_instr);
    } else {
            printf("\n%p: saw %s FALSE -- NONCE=X2M3BF73\n", offset, asm_instr);
    }
}

void view_jp_jnp(off_t offset, const char *asm_instr, uint16_t rflags)
{
    if (!(rflags & PF)) {
            printf("\n%p: saw %s TRUE -- NONCE=X2M3BF73\n", offset, asm_instr);
    } else {
            printf("\n%p: saw %s FALSE -- NONCE=X2M3BF73\n", offset, asm_instr);
    }
}
