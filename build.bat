@echo off
REM Simple build script for ML Framework on Windows

echo === ML Framework Build Script ===

REM Check for MinGW g++
where g++ >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Found g++ compiler
    goto BUILD_GCC
)

REM Check for MSVC cl
where cl >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Found MSVC cl compiler
    goto BUILD_MSVC
)

echo ERROR: No C++ compiler found!
echo Please install MinGW-w64 or Visual Studio Build Tools
echo MinGW-w64: https://www.mingw-w64.org/downloads/
echo VS Build Tools: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
pause
exit /b 1

:BUILD_GCC
echo Building with g++...
if not exist build mkdir build

g++ -std=c++11 -Wall -O3 -pthread ^
    src\ML.cpp ^
    src\Model.cpp ^
    src\Utils.cpp ^
    src\layers\Layer.cpp ^
    src\layers\Convolutional.cpp ^
    src\layers\Dense.cpp ^
    src\layers\MaxPooling.cpp ^
    src\layers\Softmax.cpp ^
    -o build\ml.exe

if %ERRORLEVEL% EQU 0 (
    echo Build successful! Run with: build\ml.exe
) else (
    echo Build failed!
    pause
    exit /b %ERRORLEVEL%
)
goto END

:BUILD_MSVC
echo Building with MSVC...
if not exist build mkdir build

cl /std:c++11 /EHsc /O2 ^
    src\ML.cpp ^
    src\Model.cpp ^
    src\Utils.cpp ^
    src\layers\Layer.cpp ^
    src\layers\Convolutional.cpp ^
    src\layers\Dense.cpp ^
    src\layers\MaxPooling.cpp ^
    src\layers\Softmax.cpp ^
    /Fe:build\ml.exe

if %ERRORLEVEL% EQU 0 (
    echo Build successful! Run with: build\ml.exe
) else (
    echo Build failed!
    pause
    exit /b %ERRORLEVEL%
)

:END
echo Done.