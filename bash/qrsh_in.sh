#! /bin/bash
USER=$(whoami)
CWD=$(dirname $0)

echo $USER:~$CWD$ qrsh -g gaa50073 -l rt_G.small=1
qrsh -g gaa50073 -l rt_G.small=1
