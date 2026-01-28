#!/bin/bash
set -e

gdown 1FBlS_Xmf6_dnCg1T0t5GSTRTwMjLuA8N
tar xvf OTPS.tar.Z

cd OTPS

for exec in extract_HC extract_local_model predict_tide
do
  ${FC} ${FCFLAGS} -o ${exec} -fconvert=swap -frecord-marker=4 ${exec}.f90 subs.f90
  cp ${exec} ${PREFIX}/bin/
done
