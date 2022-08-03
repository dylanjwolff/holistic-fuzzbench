./e9compile.sh examples/prog-stats.c
./e9tool -M 'asm == /j.*%r.*/ or asm == /call %r.*/' -P 'view(offset, asm, asm.len)@prog-stats' /usr/bin/readelf
echo ""
./a.out -a /usr/bin/readelf 2>&1 | grep NONCE | head -3
./a.out -a /usr/bin/readelf 2>&1 | grep NONCE | sort | uniq | wc -l 


