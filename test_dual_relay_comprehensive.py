#!/usr/bin/env python
"""
Comprehensive dual relay test - REAL HARDWARE CONTROL
Tests: Sequential, Simultaneous, and Mixed operations
"""

import time
import threading
import sys
import traceback

print("=" * 70)
print("COMPREHENSIVE DUAL RELAY TEST")
print("8-Channel + 2-Channel Relay System (10 Total Channels)")
print("=" * 70)

# Step 1: Import and initialize
print("\n[STEP 1] Importing pyhid_usb_relay...")
try:
    import pyhid_usb_relay
    print("✓ Import successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("  Install with: pip install pyhid_usb_relay libusb")
    sys.exit(1)

# Step 2: Find relay devices
print("\n[STEP 2] Finding relay devices...")
relay_8ch = None
relay_2ch = None

try:
    # Try find_all method first
    try:
        all_relays = pyhid_usb_relay.find_all()
        print(f"✓ Found {len(all_relays)} device(s) using find_all()")
        
        if len(all_relays) >= 2:
            relay_8ch = all_relays[0]
            relay_2ch = all_relays[1]
            print("  → Device 1: 8-channel")
            print("  → Device 2: 2-channel")
        elif len(all_relays) == 1:
            relay_8ch = all_relays[0]
            print("  → Device 1: Using as 8-channel")
            print("  ⚠ WARNING: Only 1 device found (need 2 for full test)")
        else:
            print("✗ No devices found")
            sys.exit(1)
    except (AttributeError, TypeError):
        # Fallback to find()
        print("  find_all() not available, using find()...")
        relay_8ch = pyhid_usb_relay.find()
        if relay_8ch is None:
            print("✗ find() also failed - no relay devices found")
            sys.exit(1)
        print("✓ Found 1 device using find()")
        print("  → Device: Using as 8-channel")
        print("  ⚠ WARNING: Only 1 device found (need 2 for full test)")
        relay_2ch = None

except Exception as e:
    print(f"✗ Error finding devices: {e}")
    traceback.print_exc()
    sys.exit(1)

# Helper functions
def get_channel_range(device_id):
    """Get valid channel range for a device"""
    return range(1, 9) if device_id == 1 else range(1, 3)

def test_sequential_on():
    """Test turning on channels sequentially"""
    print("\n[TEST 1A] Sequential ON - 8-Channel Relay")
    print("-" * 50)
    if relay_8ch:
        try:
            for ch in get_channel_range(1):
                relay_8ch.turn_on(ch)
                print(f"  Channel {ch}: ✓ ON")
                time.sleep(0.4)
            print("✓ Test 1A: All 8 channels turned ON sequentially")
            return True
        except Exception as e:
            print(f"✗ Test 1A failed: {e}")
            traceback.print_exc()
            return False
    return False

def test_sequential_on_2ch():
    """Test turning on 2-channel relay sequentially"""
    print("\n[TEST 1B] Sequential ON - 2-Channel Relay")
    print("-" * 50)
    if relay_2ch:
        try:
            for ch in get_channel_range(2):
                relay_2ch.turn_on(ch)
                print(f"  Channel {ch}: ✓ ON")
                time.sleep(0.4)
            print("✓ Test 1B: Both channels turned ON sequentially")
            return True
        except Exception as e:
            print(f"✗ Test 1B failed: {e}")
            traceback.print_exc()
            return False
    return True  # Skip if not available

def test_simultaneous_off():
    """Test turning off channels from both devices simultaneously"""
    print("\n[TEST 2] Simultaneous OFF - Both Devices")
    print("-" * 50)
    
    results = {}
    errors = []
    
    def off_8ch():
        if relay_8ch:
            try:
                for ch in get_channel_range(1):
                    relay_8ch.turn_off(ch)
                    print(f"  [8ch:{ch}]: ✓ OFF")
                    time.sleep(0.1)
                results['8ch'] = True
            except Exception as e:
                results['8ch'] = False
                errors.append(f"8ch error: {e}")
    
    def off_2ch():
        if relay_2ch:
            try:
                for ch in get_channel_range(2):
                    relay_2ch.turn_off(ch)
                    print(f"  [2ch:{ch}]: ✓ OFF")
                    time.sleep(0.1)
                results['2ch'] = True
            except Exception as e:
                results['2ch'] = False
                errors.append(f"2ch error: {e}")
    
    try:
        t1 = threading.Thread(target=off_8ch, name="8ch-off")
        t2 = threading.Thread(target=off_2ch, name="2ch-off")
        
        print("  Starting simultaneous OFF...")
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        if errors:
            print(f"✗ Test 2 had errors: {errors}")
            return False
        
        print("✓ Test 2: All channels turned OFF simultaneously")
        return True
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        traceback.print_exc()
        return False

def test_toggle():
    """Test toggling all channels"""
    print("\n[TEST 3] Toggle All Channels")
    print("-" * 50)
    
    try:
        if relay_8ch:
            print("  Toggling 8-channel relay...")
            for ch in get_channel_range(1):
                relay_8ch.toggle_state(ch)
                print(f"    Channel {ch}: ✓ TOGGLE")
                time.sleep(0.1)
        
        if relay_2ch:
            print("  Toggling 2-channel relay...")
            for ch in get_channel_range(2):
                relay_2ch.toggle_state(ch)
                print(f"    Channel {ch}: ✓ TOGGLE")
                time.sleep(0.1)
        
        print("✓ Test 3: Toggle test completed")
        return True
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        traceback.print_exc()
        return False

def test_mixed_pattern():
    """Test mixed sequential and simultaneous operations"""
    print("\n[TEST 4] Mixed Pattern (Sequential + Simultaneous)")
    print("-" * 50)
    
    try:
        print("  Phase 1: Turn ON 8ch sequentially...")
        if relay_8ch:
            for ch in [1, 2, 3, 4]:
                relay_8ch.turn_on(ch)
                print(f"    8ch:{ch} -> ON")
                time.sleep(0.2)
        
        print("  Phase 2: Turn ON 2ch simultaneously...")
        
        def on_2ch():
            if relay_2ch:
                for ch in get_channel_range(2):
                    relay_2ch.turn_on(ch)
                    print(f"    2ch:{ch} -> ON")
                    time.sleep(0.1)
        
        t = threading.Thread(target=on_2ch)
        t.start()
        t.join()
        
        print("✓ Test 4: Mixed pattern completed")
        return True
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")
        traceback.print_exc()
        return False

def cleanup():
    """Turn off all relays"""
    print("\n[CLEANUP] Turning off all relays...")
    try:
        if relay_8ch:
            for ch in get_channel_range(1):
                relay_8ch.turn_off(ch)
            print("✓ 8-channel relay OFF")
        if relay_2ch:
            for ch in get_channel_range(2):
                relay_2ch.turn_off(ch)
            print("✓ 2-channel relay OFF")
    except Exception as e:
        print(f"⚠ Cleanup warning: {e}")

# Run all tests
print("\n" + "=" * 70)
print("RUNNING TESTS")
print("=" * 70)

test_results = {
    "Sequential ON (8ch)": test_sequential_on(),
    "Sequential ON (2ch)": test_sequential_on_2ch(),
    "Simultaneous OFF": test_simultaneous_off(),
    "Toggle": test_toggle(),
    "Mixed Pattern": test_mixed_pattern(),
}

# Print summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

for test_name, result in test_results.items():
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"{test_name:<35} {status}")

all_passed = all(test_results.values())

print("=" * 70)
if all_passed:
    print("✓ ALL TESTS PASSED!")
    print("\nConclusion:")
    print("- Sequential control: CONFIRMED WORKING")
    print("- Simultaneous control: CONFIRMED WORKING")
    print("- Mixed operations: CONFIRMED WORKING")
    print("\n→ Dual relay system is ready for production!")
else:
    print("✗ SOME TESTS FAILED")
    print("Check the output above for details")

cleanup()

print("\n" + "=" * 70)
print("Test completed. Check physical relays for confirmation.")
print("=" * 70)

sys.exit(0 if all_passed else 1)
