./e9compile.sh examples/prog-stats.c
./e9tool -M 'mnemonic == "je" or mnemonic == "jne" or mnemonic == "jz" or mnemonic == "jnz"' -P 'view_je_jne(offset, asm, rflags)@prog-stats' /usr/bin/readelf
echo ""
./a.out -a /usr/bin/readelf 2>&1 | grep NONCE | head -3
./a.out -a /usr/bin/readelf 2>&1 | grep NONCE | sort | uniq | grep TRUE | wc -l 
./a.out -a /usr/bin/readelf 2>&1 | grep NONCE | sort | uniq | grep FALSE | wc -l 

