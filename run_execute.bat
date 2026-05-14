@echo off
cd /d C:\projetos\pairs_trading_ml

set LOG_DIR=results\logs
if not exist %LOG_DIR% mkdir %LOG_DIR%

set LOGFILE=%LOG_DIR%\execute_%date:~6,4%-%date:~3,2%-%date:~0,2%.txt

echo ===== EXECUCAO %date% %time% ===== >> %LOGFILE%
"C:\Users\55159\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe" src/execute_orders.py >> %LOGFILE% 2>&1
echo. >> %LOGFILE%
