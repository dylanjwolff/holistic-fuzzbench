set -e
e9tool -M 'mnemonic == "je" or mnemonic == "jne" or mnemonic == "jz" or mnemonic == "jnz"' -P 'view_je_jne(offset, asm, rflags)@prog-stats' $1 -o $2

