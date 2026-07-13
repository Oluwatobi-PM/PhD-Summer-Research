@echo off
SET LSHORST=no-net
@echo off
"C:\Program Files (x86)\CMG\IMEX\2009.10\Win_x64\EXE\mx200910.exe" -dd -w -log -wait -f %1 >nul
@echo off
"C:\Program Files (x86)\CMG\BR\2009.10\Win_x64\EXE\report.exe" -f ".\Data\Debug\Test_IMX\waterFlooding.rwd" -o ".\Data\Debug\Test_IMX\waterFlooding_trash.rwo" >nul
@echo off
