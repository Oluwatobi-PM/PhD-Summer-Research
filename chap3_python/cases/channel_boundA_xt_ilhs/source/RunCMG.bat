@echo off
SET CMG_LIC_HOST=License-CMG.ad.utulsa.edu
SET LSHORST=no-net
"C:\Program Files\CMG\IMEX\2025.10\Win_x64\EXE\mx202510.exe" -f %1 -wd "." -log -wait
"C:\Program Files\CMG\RESULTS\2025.10\Win_x64\exe\Report.exe" -f ".\waterFlooding.rwd" -o ".\waterFlooding.rwo"
