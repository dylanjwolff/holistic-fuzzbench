set -e

objdump -d $1 | grep "@plt" | sed 's/:.*//g' | sed 's/^[[:space:]]*//g' | grep -v '>' | sed 's/^/0x/g' | sed 's/$/,none/g' > file.csv
e9tool -M 'file[0] == offset' -P 'view(offset, asm, asm.len)@prog-stats' $1 -o $2

