set -e

e9tool -M 'asm == /j.*%r.*/ or asm == /call %r.*/' -P 'view(offset, asm, asm.len)@prog-stats' $1 -o $2


