<#
PowerShell build helper for PyInstaller on Windows.

This script attempts to locate a libusb DLL (libusb-1.0.dll / libusb-1-0.dll)
and passes it to PyInstaller with `--add-binary`. It also adds a hidden import
for `pyhid_usb_relay` so the module is included in the bundle.

Usage (from repo root):
  powershell -ExecutionPolicy Bypass -File scripts\build_dist.ps1       # creates a folder build (onedir)
  powershell -ExecutionPolicy Bypass -File scripts\build_dist.ps1 -OneFile $true  # creates a single exe (onefile)

Notes:
- If the script cannot find libusb, it will continue but warn. You can pass
  the DLL path manually by setting environment variable LIBUSB_DLL_PATH before running.
#>

param(
    [switch]$OneFile
)

function Find-LibusbDll {
    param()
    $python = (Get-Command python).Source
    # Use the helper python script to try locate the DLL
    $helper = Join-Path $PSScriptRoot "find_libusb.py"
    if (Test-Path $helper) {
        try {
            $out = & $python $helper 2>$null
            if ($out -and (Test-Path $out)) { return $out }
        } catch {
        }
    }
    # fallback: search site-packages (may be slower)
    try {
        $site = & $python -c "import site,sys; print(site.getsitepackages()[0] if site.getsitepackages() else '')" 2>$null
        if ($site) {
            $found = Get-ChildItem -Path $site -Filter "libusb-1*.dll" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($found) { return $found.FullName }
        }
    } catch {
    }
    # last resort: search user profile (may be slow)
    $maybe = Get-ChildItem -Path $env:USERPROFILE -Filter "libusb-1*.dll" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($maybe) { return $maybe.FullName }
    return ""
}

Write-Host "Preparing PyInstaller build (OneFile=$OneFile)" -ForegroundColor Cyan

$dllPath = $env:LIBUSB_DLL_PATH
if (-not $dllPath) {
    $dllPath = Find-LibusbDll
}

if ($dllPath) {
    Write-Host "Found libusb DLL: $dllPath" -ForegroundColor Green
} else {
    Write-Warning "libusb DLL not found automatically. If your app needs libusb (pyhid_usb_relay), set LIBUSB_DLL_PATH env var to the DLL path or place the DLL next to the exe after building."
}

# Build args
$pyinstaller = "pyinstaller"
$args = @()
if ($OneFile) { $args += "--onefile" } else { $args += "--onedir" }
$args += "--name"; $args += "Pokayoke"
$args += "--hidden-import"; $args += "pyhid_usb_relay"
# optionally include other hidden imports if needed
# add the DLL as a binary so it's extracted into the bundle
if ($dllPath) {
    $args += "--add-binary"; $args += "$dllPath;."
}
# entry script (adjust if your main entry file differs)
$args += "human_hikvision.py"

Write-Host "Running: $pyinstaller $($args -join ' ')" -ForegroundColor Cyan
& $pyinstaller @args

if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller failed with code $LASTEXITCODE"
    exit $LASTEXITCODE
} else {
    Write-Host "PyInstaller finished. Check the 'dist' directory." -ForegroundColor Green
}
