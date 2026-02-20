"""Small helper to locate libusb DLL files in the current Python environment.
Prints the full path to the first matching DLL or an empty string if none found.

Usage: python scripts/find_libusb.py
"""
import importlib
import os
import glob
import site

def find_in_module(mod_name: str):
    try:
        mod = importlib.import_module(mod_name)
        base = os.path.dirname(mod.__file__)
        # search recursively for libusb DLL variants
        patterns = ["libusb-1.0*.dll", "libusb-1-0*.dll"]
        for pat in patterns:
            matches = glob.glob(os.path.join(base, "**", pat), recursive=True)
            if matches:
                return matches[0]
    except Exception:
        return ""
    return ""

def find_in_sitepackages():
    try:
        dirs = site.getsitepackages()
    except Exception:
        dirs = [site.getusersitepackages()]
    patterns = ["libusb-1.0*.dll", "libusb-1-0*.dll"]
    for d in dirs:
        for pat in patterns:
            matches = glob.glob(os.path.join(d, "**", pat), recursive=True)
            if matches:
                return matches[0]
    return ""

if __name__ == '__main__':
    # try to find under pyhid_usb_relay module first
    p = find_in_module('pyhid_usb_relay')
    if not p:
        # try other likely modules
        p = find_in_module('libusb1') or find_in_sitepackages()
    print(p or "")
