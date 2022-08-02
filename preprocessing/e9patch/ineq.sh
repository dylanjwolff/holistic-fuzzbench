./e9compile.sh examples/prog-stats.c
./e9tool -M 'mnemonic == "jnle" or mnemonic == "jle" or mnemonic == "jnl" or mnemonic == "jl" or mnemonic == "jnp" or mnemonic == "jp" or mnemonic == "js" or mnemonic == "jns" or mnemonic == "jnbe" or mnemonic == "jbe" or mnemonic == "jnb" or mnemonic == "jb"' -P 'view(offset, asm, asm.len)@prog-stats' /usr/bin/readelf
echo ""
./a.out -a /usr/bin/readelf 2>&1 | grep NONCE | head -3
./a.out -a /usr/bin/readelf 2>&1 | grep NONCE | wc -l 

