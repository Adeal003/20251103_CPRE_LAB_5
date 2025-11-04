# PowerShell Build Script for ML Framework
# Compatible with Windows PowerShell and MinGW-w64

param(
    [string]$Config = "Release",
    [switch]$Clean,
    [switch]$Help
)

if ($Help) {
    Write-Host "ML Framework Build Script"
    Write-Host "Usage: .\build.ps1 [-Config Release|Debug] [-Clean] [-Help]"
    Write-Host ""
    Write-Host "Parameters:"
    Write-Host "  -Config    Build configuration (Release or Debug, default: Release)"
    Write-Host "  -Clean     Clean build directory before building"
    Write-Host "  -Help      Show this help message"
    Write-Host ""
    Write-Host "Requirements:"
    Write-Host "  - MinGW-w64 with g++ available in PATH"
    Write-Host "  - Or Visual Studio Build Tools with cl.exe"
    exit 0
}

# Configuration
$SourceDir = "src"
$BuildDir = "build"
$ExeName = "ml.exe"
$ExePath = Join-Path $BuildDir $ExeName

# Compiler settings
$CppStandard = "c++11"
$WarningFlags = @("-Wall", "-Werror", "-pedantic")
$OptFlags = if ($Config -eq "Debug") { @("-g", "-Og") } else { @("-O3") }
$LinkFlags = @("-pthread")

# Find all source files
$SourceFiles = Get-ChildItem -Recurse -Path $SourceDir -Include "*.cpp" | ForEach-Object { $_.FullName }

Write-Host "=== ML Framework Build Script ===" -ForegroundColor Cyan
Write-Host "Configuration: $Config" -ForegroundColor Green
Write-Host "Source files found: $($SourceFiles.Count)" -ForegroundColor Green

# Clean if requested
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BuildDir
}

# Create build directory
if (!(Test-Path $BuildDir)) {
    Write-Host "Creating build directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# Check for compiler
$Compiler = $null
$CompilerArgs = @()

# Try g++ first (MinGW)
try {
    $null = Get-Command "g++" -ErrorAction Stop
    $Compiler = "g++"
    $CompilerArgs = @("-std=$CppStandard") + $WarningFlags + $OptFlags + $SourceFiles + @("-o", $ExePath) + $LinkFlags
    Write-Host "Using g++ compiler" -ForegroundColor Green
} catch {
    # Try cl.exe (MSVC)
    try {
        $null = Get-Command "cl" -ErrorAction Stop
        $Compiler = "cl"
        # MSVC has different flag syntax
        $CompilerArgs = @("/std:c++11", "/EHsc") + $SourceFiles + @("/Fe:$ExePath")
        if ($Config -eq "Debug") {
            $CompilerArgs += @("/Zi", "/Od")
        } else {
            $CompilerArgs += @("/O2")
        }
        Write-Host "Using MSVC cl compiler" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: No suitable C++ compiler found!" -ForegroundColor Red
        Write-Host "Please install one of the following:" -ForegroundColor Yellow
        Write-Host "  - MinGW-w64: https://www.mingw-w64.org/downloads/" -ForegroundColor Yellow
        Write-Host "  - Visual Studio Build Tools: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022" -ForegroundColor Yellow
        exit 1
    }
}

# Compile
Write-Host "Compiling..." -ForegroundColor Yellow
Write-Host "Command: $Compiler $($CompilerArgs -join ' ')" -ForegroundColor DarkGray

try {
    & $Compiler @CompilerArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Build successful! Executable: $ExePath" -ForegroundColor Green
        Write-Host "Run with: .\$ExePath" -ForegroundColor Cyan
    } else {
        Write-Host "Build failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} catch {
    Write-Host "Build failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}