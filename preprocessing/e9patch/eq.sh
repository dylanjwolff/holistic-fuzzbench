./e9compile.sh examples/prog-stats.c
./e9tool -M 'mnemonic == "jz" or mnemonic == "jnz"' -P 'view(offset, asm, asm.len)@prog-stats' /usr/bin/readelf
echo ""
./a.out -a /usr/bin/readelf 2>&1 | grep NONCE | head -3
./a.out -a /usr/bin/readelf 2>&1 | grep NONCE | wc -l 

