set -e

./e9compile.sh examples/prog-stats.c
./e9tool  -M 'mnemonic == "jg" or mnemonic == "jnle" or mnemonic == "jle" or mnemonic == "jng"' -P 'view_jle_jg(offset, asm, rflags)@prog-stats' \
  -M 'mnemonic == "jl" or mnemonic == "jnge" or mnemonic == "jge" or mnemonic == "jnl"' -P 'view_jl_jge(offset, asm, rflags)@prog-stats' \
  -M 'mnemonic == "jbe" or mnemonic == "jna" or mnemonic == "ja" or mnemonic == "jnbe"' -P 'view_jbe_ja(offset, asm, rflags)@prog-stats' \
  -M 'mnemonic == "jb" or mnemonic == "jnae" or mnemonic == "jc" or mnemonic == "jnb" or mnemonic == "jae" or mnemonic = "jnc"' -P 'view_jb_jnb(offset, asm, rflags)@prog-stats' \
  -M 'mnemonic == "js" or mnemonic == "jns"' -P 'view_js_jns(offset, asm, rflags)@prog-stats' \
        /usr/bin/readelf

echo ""
./a.out -a /usr/bin/readelf 2>&1 | grep NONCE | head -3
./a.out -a /usr/bin/readelf 2>&1 | grep NONCE | sort | uniq | grep TRUE | wc -l 
./a.out -a /usr/bin/readelf 2>&1 | grep NONCE | sort | uniq | grep FALSE | wc -l 

