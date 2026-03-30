#!/usr/bin/env python
"""
Real Hardware Test - Dual Relay Control
Tests actual physical relay switching
"""

import time
import threading
import sys

print("=" * 70)
print("REAL HARDWARE RELAY TEST")
print("=" * 70)

# Import library
try:
    import pyhid_usb_relay
    print("\n✓ pyhid_usb_relay imported")
except ImportError:
    print("✗ pyhid_usb_relay not installed")
    print("  Install with: pip install pyhid_usb_relay libusb")
    sys.exit(1)

# Find relay device
print("\n[STEP 1] Finding relay device...")
try:
    relay = pyhid_usb_relay.find()
    if relay is None:
        print("✗ No relay device found!")
        sys.exit(1)
    
    print(f"✓ Relay device found!")
    print(f"  Type: {type(relay)}")
    
    # Get device info
    try:
        print(f"  Serial: {relay.serial}")
    except:
        pass
    try:
        print(f"  Product: {relay.product}")
    except:
        pass
    try:
        print(f"  Num relays: {relay.num_relays}")
    except:
        pass
    try:
        print(f"  Current state: {relay.state}")
    except:
        pass

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test functions
print("\n" + "=" * 70)
print("TEST 1: Sequential Turn ON")
print("=" * 70)

try:
    print("\nTurning ON channels 1-8 sequentially...")
    for ch in range(1, 9):
        relay.turn_on(ch)
        print(f"  Channel {ch}: ON ✓")
        time.sleep(0.3)
    print("\n✓ All 8 channels ON (sequential)")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

time.sleep(1)

print("\n" + "=" * 70)
print("TEST 2: Sequential Turn OFF")
print("=" * 70)

try:
    print("\nTurning OFF channels 1-8 sequentially...")
    for ch in range(1, 9):
        relay.turn_off(ch)
        print(f"  Channel {ch}: OFF ✓")
        time.sleep(0.3)
    print("\n✓ All 8 channels OFF (sequential)")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

time.sleep(1)

print("\n" + "=" * 70)
print("TEST 3: Toggle Pattern")
print("=" * 70)

try:
    print("\nToggling channels 1-8...")
    for ch in range(1, 9):
        relay.toggle_state(ch)
        print(f"  Channel {ch}: TOGGLE ✓")
        time.sleep(0.2)
    print("\n✓ All channels toggled")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

time.sleep(0.5)

print("\n" + "=" * 70)
print("TEST 4: Rapid Toggle (Stress Test)")
print("=" * 70)

try:
    print("\nRapid toggling for 3 cycles...")
    for cycle in range(3):
        print(f"  Cycle {cycle + 1}:", end=" ")
        for ch in range(1, 9):
            relay.toggle_state(ch)
        print("✓")
        time.sleep(0.3)
    print("\n✓ Stress test completed")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
print("\n" + "=" * 70)
print("CLEANUP: Turning OFF all channels")
print("=" * 70)

try:
    for ch in range(1, 9):
        relay.turn_off(ch)
    print("✓ All channels OFF")
except Exception as e:
    print(f"⚠ Warning: {e}")

print("\n" + "=" * 70)
print("✓ TEST COMPLETED")
print("=" * 70)
print("\nIf you saw the relays switching physically:")
print("  ✓ Sequential control is WORKING")
print("  ✓ Toggle control is WORKING")
print("  ✓ Relay hardware is responding correctly")
print("\nYou can now test with dual relays!")
print("=" * 70)
