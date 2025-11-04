@echo off
REM Build script that activates Visual Studio environment automatically

echo === ML Framework Build Script with VS Build Tools ===

REM Activate Visual Studio Build Tools environment
echo Activating Visual Studio Build Tools environment...
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -no_logo

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate Visual Studio environment
    pause
    exit /b 1
)

REM Check if cl is now available
where cl >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: MSVC compiler not found after environment activation
    pause
    exit /b 1
)

echo MSVC compiler found!
echo Building with MSVC...

REM Create build directory
if not exist build mkdir build

REM Compile with MSVC
cl /std:c++14 /EHsc /O2 /nologo ^
    src\ML.cpp ^
    src\Model.cpp ^
    src\Utils.cpp ^
    src\layers\Layer.cpp ^
    src\layers\Convolutional.cpp ^
    src\layers\Dense.cpp ^
    src\layers\Flatten.cpp ^
    src\layers\MaxPooling.cpp ^
    src\layers\Softmax.cpp ^
    /Fe:build\ml.exe

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Build successful! 
    echo Executable: build\ml.exe
    echo ========================================
    echo.
    echo Run with: build\ml.exe
) else (
    echo.
    echo ========================================
    echo Build failed!
    echo ========================================
    pause
    exit /b %ERRORLEVEL%
)

echo Done.
pause