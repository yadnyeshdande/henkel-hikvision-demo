#!/usr/bin/env python
"""Simple test script for dual relay control - REAL HARDWARE"""

import time
import threading
import sys

print("=" * 60)
print("DUAL RELAY TEST - 8ch + 2ch = 10 channels total")
print("=" * 60)

# Import and initialize real relays
try:
    import pyhid_usb_relay
    print("\n✓ pyhid_usb_relay imported successfully")
    
    # Try to find all connected relay devices
    try:
        # Method 1: Try find_all
        all_relays = pyhid_usb_relay.find_all()
        print(f"  Found {len(all_relays)} relay device(s)")
        
        if len(all_relays) >= 2:
            relay_8ch = all_relays[0]
            relay_2ch = all_relays[1]
            print(f"  Using relay 0 as 8-channel, relay 1 as 2-channel")
        elif len(all_relays) == 1:
            relay_8ch = all_relays[0]
            relay_2ch = None
            print(f"  Found only 1 relay device - using for 8-channel only")
        else:
            raise Exception("No relay devices found")
    except AttributeError:
        # Method 2: Use find (single device)
        relay_8ch = pyhid_usb_relay.find()
        relay_2ch = None
        print(f"  Using find() - found 1 relay device")
        if relay_8ch is None:
            print("  ✗ No relay device found!")
            sys.exit(1)
    
    print(f"  Relay 8ch type: {type(relay_8ch)}")
    if relay_2ch:
        print(f"  Relay 2ch type: {type(relay_2ch)}")
    
except ImportError as e:
    print(f"✗ Cannot import pyhid_usb_relay: {e}")
    print("  Make sure to install: pip install pyhid_usb_relay libusb")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error initializing relays: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[TEST 1] Sequential Control")
print("-" * 40)
print("Turning ON channels 1-8 sequentially (8-channel relay)...")

if relay_8ch:
    try:
        for i in range(1, 9):
            relay_8ch.turn_on(i)
            print(f"  8ch:{i} -> ON")
            time.sleep(0.3)
        print("✓ All 8 channels turned ON sequentially")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if relay_2ch:
    print("\nTurning ON channels 1-2 sequentially (2-channel relay)...")
    try:
        for i in range(1, 3):
            relay_2ch.turn_on(i)
            print(f"  2ch:{i} -> ON")
            time.sleep(0.3)
        print("✓ Both 2-channel relays turned ON sequentially")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

time.sleep(1)

print("\n[TEST 2] Simultaneous Control")
print("-" * 40)
print("Turning OFF all channels simultaneously...")

def turn_off_8ch():
    if relay_8ch:
        for i in range(1, 9):
            try:
                relay_8ch.turn_off(i)
                print(f"  8ch:{i} -> OFF")
                time.sleep(0.1)
            except Exception as e:
                print(f"  Error turning off 8ch:{i}: {e}")

def turn_off_2ch():
    if relay_2ch:
        for i in range(1, 3):
            try:
                relay_2ch.turn_off(i)
                print(f"  2ch:{i} -> OFF")
                time.sleep(0.1)
            except Exception as e:
                print(f"  Error turning off 2ch:{i}: {e}")

t1 = threading.Thread(target=turn_off_8ch)
t2 = threading.Thread(target=turn_off_2ch)

t1.start()
t2.start()
t1.join()
t2.join()

print("✓ All channels turned OFF simultaneously")

time.sleep(1)

print("\n[TEST 3] Toggle Pattern")
print("-" * 40)
print("Toggling all channels...")

if relay_8ch:
    try:
        for i in range(1, 9):
            relay_8ch.toggle_state(i)
            print(f"  8ch:{i} -> TOGGLE")
            time.sleep(0.1)
    except Exception as e:
        print(f"✗ Error: {e}")

if relay_2ch:
    try:
        for i in range(1, 3):
            relay_2ch.toggle_state(i)
            print(f"  2ch:{i} -> TOGGLE")
            time.sleep(0.1)
    except Exception as e:
        print(f"✗ Error: {e}")

print("✓ Toggle pattern completed")

time.sleep(0.5)

# Final cleanup - turn everything OFF
print("\n[CLEANUP] Final OFF sequence...")
print("-" * 40)

if relay_8ch:
    try:
        for i in range(1, 9):
            relay_8ch.turn_off(i)
        print("✓ 8-channel relay all OFF")
    except Exception as e:
        print(f"✗ Error: {e}")

if relay_2ch:
    try:
        for i in range(1, 3):
            relay_2ch.turn_off(i)
        print("✓ 2-channel relay all OFF")
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("✓ TEST COMPLETED - Check physical relays for toggling")
print("=" * 60)
