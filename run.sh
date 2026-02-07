#!/usr/bin/env fish

printf "> JVBERi0xLjUNJeLjz9MNCjM0IDAgb2JqDTw8L0xpbmVhcml6ZWQgMS9MIDI3NjAyOC9PIDM2L0Ug\n" > page-000.txt

# parallel -j 16 uv.exe run ./cluster.py ../EFTA00400459-{}_2x.png -o page-{}.txt ::: (printf "%03d\n" (seq 1 74))
# uv.exe run ./cluster.py ../EFTA00400459-075_2x.png -o page-075.txt --lines 34

# parallel -j 16 uv.exe run ./cluster.py -d ../EFTA00400459-{}.png -o npage-{}.txt ::: (printf "%03d\n" (seq 1 75))
parallel -j 16 uv.exe run ./cluster.py -d ../EFTA00400459-{}_2x.png -o page-{}.txt ::: (printf "%03d\n" (seq 1 75))
# Buffer output to allow replacing input without `sponge`
string replace -m1 -r "\+\+.*" "==" -- (cat page-075.txt) > page-075.txt
