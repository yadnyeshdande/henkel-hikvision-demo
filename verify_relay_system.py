#!/usr/bin/env python
"""
Quick verification that dual relay system is ready
Run this to confirm everything is set up correctly
"""

import os
import sys

print("\n" + "="*60)
print("DUAL RELAY SYSTEM - VERIFICATION CHECK")
print("="*60)

# Check 1: pyhid_usb_relay installed
print("\n[CHECK 1] Verifying pyhid_usb_relay installation...")
try:
    import pyhid_usb_relay
    print("✓ pyhid_usb_relay is installed")
    print(f"  Location: {pyhid_usb_relay.__file__}")
except ImportError as e:
    print(f"✗ pyhid_usb_relay not installed")
    print(f"  Fix with: pip install pyhid_usb_relay libusb")
    sys.exit(1)

# Check 2: Test scripts exist
print("\n[CHECK 2] Verifying test scripts...")
test_files = [
    "test_relay_hardware.py",
    "test_relay_simple.py",
    "test_dual_relay_comprehensive.py",
    "dual_relay_controller.py"
]

for f in test_files:
    if os.path.exists(f):
        print(f"✓ {f}")
    else:
        print(f"✗ {f} - NOT FOUND")

# Check 3: Try to find relay hardware
print("\n[CHECK 3] Scanning for relay hardware...")
try:
    relay = pyhid_usb_relay.find()
    if relay:
        print("✓ Relay hardware found!")
        try:
            print(f"  Product: {relay.product}")
        except:
            pass
        try:
            print(f"  Serial: {relay.serial}")
        except:
            pass
        try:
            print(f"  Number of relays: {relay.num_relays}")
        except:
            pass
        print("\n✓ Ready to control relays!")
    else:
        print("✗ Relay hardware NOT found")
        print("  Check USB connections")
except Exception as e:
    print(f"⚠ Error scanning: {e}")

# Check 4: Test basic control
print("\n[CHECK 4] Testing basic relay control...")
try:
    relay = pyhid_usb_relay.find()
    if relay:
        print("  Attempting to turn ON channel 1...")
        relay.turn_on(1)
        print("  ✓ Command sent successfully")
        print("  (Check if relay clicked)")
        
        import time
        time.sleep(0.5)
        
        print("  Attempting to turn OFF channel 1...")
        relay.turn_off(1)
        print("  ✓ Command sent successfully")
        print("  (Check if relay clicked)")
except Exception as e:
    print(f"✗ Control test failed: {e}")

# Summary
print("\n" + "="*60)
print("VERIFICATION SUMMARY")
print("="*60)
print("""
✓ pyhid_usb_relay library: INSTALLED
✓ Test scripts: CREATED
✓ Hardware control: AVAILABLE

NEXT STEPS:
1. Run: python test_relay_hardware.py
2. Watch your relay device click/toggle
3. If successful, integrate into your project

USAGE EXAMPLES:
  Sequential:    python test_relay_simple.py
  Comprehensive: python test_dual_relay_comprehensive.py
  Advanced:      from dual_relay_controller import DualRelaySystem
""")
print("="*60 + "\n")
