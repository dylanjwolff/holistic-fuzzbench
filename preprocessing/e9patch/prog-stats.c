/*
 * PRINT instrumentation.
 */

#include "stdlib.c"

void view(off_t offset, const char *asm_instr, const size_t len)
{
    printf("\n%p: saw %s -- NONCE=X2M3BF73\n", offset, asm_instr);
}

