@echo off
REM Run this script as Administrator

setlocal EnableDelayedExpansion

REM Define CUDA install path here
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"

echo Adding CUDA paths to system environment variables...

REM Query current system PATH
for /f "skip=2 tokens=2,*" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path') do (
    set "CurrentPath=%%B"
)

REM Remove trailing spaces if any
set "CurrentPath=!CurrentPath:~0,-1!"

REM Compose new PATH with CUDA paths appended
set "NewPath=!CurrentPath!;%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp"

REM Show new PATH to user for confirmation
echo New PATH will be:
echo !NewPath!

REM Set new system PATH (use /M to modify system environment variables)
setx PATH "!NewPath!" /M

echo CUDA environment variables have been updated.
echo Please restart your command prompt or PC for changes to take effect.
pause
