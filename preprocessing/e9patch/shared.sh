./e9compile.sh examples/prog-stats.c
objdump -d /usr/bin/readelf | grep "@plt" | sed 's/:.*//g' | sed 's/^[[:space:]]*//g' | grep -v '>' | sed 's/^/0x/g' | sed 's/$/,none/g' > file.csv
./e9tool -M 'file[0] == offset' -P 'view(offset, asm, asm.len)@prog-stats' /usr/bin/readelf
echo ""
./a.out -a /usr/bin/readelf 2>&1 | grep NONCE | head -3
./a.out -a /usr/bin/readelf 2>&1 | grep NONCE | sort | uniq | wc -l 

