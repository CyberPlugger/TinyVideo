@echo off
title TinyVideo

echo Checking Python interpreter...

"C:\Users\dikid\PycharmProjects\someshit\TinyVideo\.venv\Scripts\python.exe" -c "import struct; print(struct.calcsize('P') * 8)" > "%TEMP%\tinyvideo_python_bits.txt" 2>nul

if errorlevel 1 (
    echo.
    echo Python could not be started.
    echo.
    pause
    exit /b
)

set /p BITS=<"%TEMP%\tinyvideo_python_bits.txt"

if "%BITS%"=="64" (
    echo 64-bit Python detected.
    echo Starting TinyVideo...
    echo.
    "C:\Users\dikid\PycharmProjects\someshit\TinyVideo\.venv\Scripts\python.exe" "C:\Users\dikid\PycharmProjects\someshit\TinyVideo\main.py"
) else (
    echo.
    echo Your Python interpreter is 32-bit.
    echo Please install a 64-bit version of Python.
    echo.
    pause
)

del "%TEMP%\tinyvideo_python_bits.txt" >nul 2>&1
