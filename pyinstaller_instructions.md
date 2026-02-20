# Building the app with PyInstaller (include libusb for `pyhid_usb_relay`)

This project optionally uses `pyhid_usb_relay`, which depends on a libusb runtime DLL (e.g. `libusb-1.0.dll`). When bundling with PyInstaller on Windows you must ensure that DLL is included in the final distribution.

Files added:
- `scripts/find_libusb.py` — helper to locate libusb DLL in the current Python environment.
- `scripts/build_dist.ps1` — PowerShell build helper that runs PyInstaller and includes the DLL via `--add-binary` and `--hidden-import pyhid_usb_relay`.

Quick steps (Windows, from repo root):

1. Install build deps in your virtualenv:

```powershell
python -m pip install -U pyinstaller pyhid_usb_relay
```

2. Run the build script (onedir output - easier to debug):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_dist.ps1
```

Or build a single-file EXE (note: onefile bundles extract at runtime which can change where DLLs end up):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_dist.ps1 -OneFile
```

3. If the script cannot find the libusb DLL automatically, find it on your system (for example inside the `pyhid_usb_relay` site-packages or installed libusb package) and set `LIBUSB_DLL_PATH` before running:

```powershell
$env:LIBUSB_DLL_PATH = 'C:\path\to\libusb-1.0.dll'
powershell -ExecutionPolicy Bypass -File scripts\build_dist.ps1
```

Troubleshooting
- If you get a runtime error about `libusb-1.0.dll` not found when running the bundled exe, either:
  - Re-run the build and ensure the DLL is included (check `dist\Pokayoke\` or the unpacked folder inside the onefile runtime temp dir).
  - Place `libusb-1.0.dll` next to the final exe (works for onedir) or set `LIBUSB_DLL_PATH` and rebuild.

- Hidden imports: if PyInstaller misses other imports from `pyhid_usb_relay`, add additional `--hidden-import` flags to `scripts\build_dist.ps1`.

- OneFile caveat: onefile mode creates a single exe — native DLLs are extracted into a temporary folder at runtime. Some libraries may expect the DLL to be co-located with the exe; if you have trouble, use `--onedir` for debugging and then test `--onefile`.

Notes for packaging and distribution
- Test the built app on a clean Windows machine (or VM) that does not have libusb installed to verify bundling.
- If you need to sign the executable, sign the final exe after building.

If you want, I can:
- run a quick search to try to locate `libusb-1.0.dll` on your current environment,
- or update `human_hikvision.py` to provide a clearer runtime error when the DLL is missing.
