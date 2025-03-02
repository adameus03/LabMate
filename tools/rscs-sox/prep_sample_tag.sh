#!/bin/bash

cd /mnt/rscs;
MYSID=$(cat uhfd/sid); echo -n $MYSID > uhfd/mkdev; cd "uhf$(cat uhfd/result/$MYSID/value)";
echo -n 01020304 > access_passwd;
echo -n 04030201 > kill_passwd;
echo -n 434343434343434343434343 > epc;
echo -n 02 > flags;
cd -;
