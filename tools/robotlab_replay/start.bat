@echo off
setlocal
set PYTHONUTF8=1
py -3 "%~dp0server.py" --open %*
endlocal
