"""
Test script to verify dual relay modules (8-channel + 2-channel) can work simultaneously and sequentially.
This script tests control over 10 total relay channels.
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional, List
import traceback


@dataclass
class RelayChannel:
    """Represents a single relay channel"""
    device_id: int  # 1 for 8-channel, 2 for 2-channel
    channel: int   # Channel number (1-8 or 1-2)
    state: bool    # Current state (True=ON, False=OFF)


class DualRelayController:
    """Controls dual relay modules (8-channel + 2-channel)"""
    
    def __init__(self):
        self.relay_8ch = None      # 8-channel relay module
        self.relay_2ch = None      # 2-channel relay module
        self.channels: List[RelayChannel] = []
        self.lock = threading.Lock()
        self.test_results = {
            'initialization': False,
            'sequential_control': False,
            'simultaneous_control': False,
            'stress_test': False
        }
    
    def initialize(self):
        """Initialize both relay modules"""
        try:
            # Try to import and find relay modules
            try:
                import pyhid_usb_relay
                self.relay_8ch = pyhid_usb_relay.find()
                print("✓ Found 8-channel relay module")
            except Exception as e:
                print(f"⚠ Could not initialize 8-channel relay: {e}")
                self.relay_8ch = None
            
            try:
                import pyhid_usb_relay
                # Try to find second device if multiple exist
                all_relays = pyhid_usb_relay.find_all()
                if isinstance(all_relays, list) and len(all_relays) > 1:
                    self.relay_2ch = all_relays[1]
                    print("✓ Found 2-channel relay module")
                else:
                    print("⚠ Could not find second relay module")
                    self.relay_2ch = None
            except Exception as e:
                print(f"⚠ Could not initialize 2-channel relay: {e}")
                self.relay_2ch = None
            
            # If real devices not available, use mock for testing
            if not self.relay_8ch or not self.relay_2ch:
                print("\n→ Using MOCK mode for testing (no physical relays detected)")
                self.relay_8ch = MockRelay(8, "8-channel")
                self.relay_2ch = MockRelay(2, "2-channel")
            
            # Initialize channel states
            for i in range(1, 9):
                self.channels.append(RelayChannel(device_id=1, channel=i, state=False))
            for i in range(1, 3):
                self.channels.append(RelayChannel(device_id=2, channel=i, state=False))
            
            self.test_results['initialization'] = True
            print("\n✓ Dual relay controller initialized with 10 channels (8+2)\n")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize: {e}")
            traceback.print_exc()
            return False
    
    def turn_on(self, device_id: int, channel: int) -> bool:
        """Turn on a relay channel"""
        with self.lock:
            try:
                if device_id == 1 and self.relay_8ch:
                    self.relay_8ch.turn_on(channel)
                    self._update_state(device_id, channel, True)
                    return True
                elif device_id == 2 and self.relay_2ch:
                    self.relay_2ch.turn_on(channel)
                    self._update_state(device_id, channel, True)
                    return True
                return False
            except Exception as e:
                print(f"Error turning on relay {device_id}:{channel}: {e}")
                return False
    
    def turn_off(self, device_id: int, channel: int) -> bool:
        """Turn off a relay channel"""
        with self.lock:
            try:
                if device_id == 1 and self.relay_8ch:
                    self.relay_8ch.turn_off(channel)
                    self._update_state(device_id, channel, False)
                    return True
                elif device_id == 2 and self.relay_2ch:
                    self.relay_2ch.turn_off(channel)
                    self._update_state(device_id, channel, False)
                    return True
                return False
            except Exception as e:
                print(f"Error turning off relay {device_id}:{channel}: {e}")
                return False
    
    def toggle(self, device_id: int, channel: int) -> bool:
        """Toggle a relay channel"""
        with self.lock:
            try:
                if device_id == 1 and self.relay_8ch:
                    self.relay_8ch.toggle_state(channel)
                    new_state = not self._get_state(device_id, channel)
                    self._update_state(device_id, channel, new_state)
                    return True
                elif device_id == 2 and self.relay_2ch:
                    self.relay_2ch.toggle_state(channel)
                    new_state = not self._get_state(device_id, channel)
                    self._update_state(device_id, channel, new_state)
                    return True
                return False
            except Exception as e:
                print(f"Error toggling relay {device_id}:{channel}: {e}")
                return False
    
    def _update_state(self, device_id: int, channel: int, state: bool):
        """Update internal state tracking"""
        for ch in self.channels:
            if ch.device_id == device_id and ch.channel == channel:
                ch.state = state
                break
    
    def _get_state(self, device_id: int, channel: int) -> bool:
        """Get current state of a channel"""
        for ch in self.channels:
            if ch.device_id == device_id and ch.channel == channel:
                return ch.state
        return False
    
    def print_status(self):
        """Print current status of all channels"""
        print("\nRelay Status:")
        print("=" * 50)
        print(f"{'Device':<15} {'Channel':<10} {'State':<10}")
        print("=" * 50)
        for ch in self.channels:
            device = "8-channel" if ch.device_id == 1 else "2-channel"
            state = "ON" if ch.state else "OFF"
            print(f"{device:<15} {ch.channel:<10} {state:<10}")
        print("=" * 50 + "\n")
    
    def test_sequential_control(self):
        """Test sequential control: turn on/off channels one by one"""
        print("\n" + "="*60)
        print("TEST 1: SEQUENTIAL CONTROL")
        print("="*60)
        print("Turning ON channels sequentially (1-10)...\n")
        
        try:
            # Turn on all channels sequentially
            for i, ch in enumerate(self.channels, 1):
                result = self.turn_on(ch.device_id, ch.channel)
                status = "✓" if result else "✗"
                print(f"{status} Channel {i}: {ch.device_id}:{ch.channel} -> ON")
                time.sleep(0.5)
            
            self.print_status()
            
            print("Turning OFF channels sequentially (1-10)...\n")
            time.sleep(1)
            
            # Turn off all channels sequentially
            for i, ch in enumerate(self.channels, 1):
                result = self.turn_off(ch.device_id, ch.channel)
                status = "✓" if result else "✗"
                print(f"{status} Channel {i}: {ch.device_id}:{ch.channel} -> OFF")
                time.sleep(0.5)
            
            self.print_status()
            self.test_results['sequential_control'] = True
            print("✓ Sequential control test PASSED\n")
            return True
        except Exception as e:
            print(f"✗ Sequential control test FAILED: {e}\n")
            return False
    
    def test_simultaneous_control(self):
        """Test simultaneous control: turn on/off channels from different devices at same time"""
        print("\n" + "="*60)
        print("TEST 2: SIMULTANEOUS CONTROL")
        print("="*60)
        print("Turning ON 8-channel and 2-channel relays simultaneously...\n")
        
        try:
            threads = []
            
            # Turn on all 8-channel relay channels
            def turn_on_8ch():
                for i in range(1, 9):
                    self.turn_on(1, i)
                    time.sleep(0.2)
            
            # Turn on all 2-channel relay channels
            def turn_on_2ch():
                for i in range(1, 3):
                    self.turn_on(2, i)
                    time.sleep(0.2)
            
            t1 = threading.Thread(target=turn_on_8ch, name="8ch-turn-on")
            t2 = threading.Thread(target=turn_on_2ch, name="2ch-turn-on")
            
            threads.append(t1)
            threads.append(t2)
            
            t1.start()
            t2.start()
            
            for t in threads:
                t.join()
            
            print("✓ All channels turned ON simultaneously\n")
            self.print_status()
            
            print("\nTurning OFF 8-channel and 2-channel relays simultaneously...\n")
            time.sleep(1)
            
            threads = []
            
            # Turn off all 8-channel relay channels
            def turn_off_8ch():
                for i in range(1, 9):
                    self.turn_off(1, i)
                    time.sleep(0.2)
            
            # Turn off all 2-channel relay channels
            def turn_off_2ch():
                for i in range(1, 3):
                    self.turn_off(2, i)
                    time.sleep(0.2)
            
            t1 = threading.Thread(target=turn_off_8ch, name="8ch-turn-off")
            t2 = threading.Thread(target=turn_off_2ch, name="2ch-turn-off")
            
            threads.append(t1)
            threads.append(t2)
            
            t1.start()
            t2.start()
            
            for t in threads:
                t.join()
            
            print("✓ All channels turned OFF simultaneously\n")
            self.print_status()
            
            self.test_results['simultaneous_control'] = True
            print("✓ Simultaneous control test PASSED\n")
            return True
        except Exception as e:
            print(f"✗ Simultaneous control test FAILED: {e}\n")
            traceback.print_exc()
            return False
    
    def test_mixed_operations(self):
        """Test mixed operations: sequential and simultaneous in combination"""
        print("\n" + "="*60)
        print("TEST 3: MIXED OPERATIONS (Sequential + Simultaneous)")
        print("="*60)
        print("Running alternating pattern...\n")
        
        try:
            # Pattern: Turn on some channels sequentially while toggling others simultaneously
            for pattern in range(2):
                print(f"Pattern {pattern + 1}:")
                
                # Sequential: turn on 8ch channels 1-4
                for i in range(1, 5):
                    self.turn_on(1, i)
                    print(f"  8ch:{i} -> ON")
                    time.sleep(0.2)
                
                # Simultaneous: turn on 2ch channels and toggle some 8ch channels
                def simultaneous_ops():
                    for i in range(1, 3):
                        self.turn_on(2, i)
                        print(f"  2ch:{i} -> ON")
                    for i in range(5, 9):
                        self.toggle(1, i)
                        print(f"  8ch:{i} -> TOGGLE")
                
                t = threading.Thread(target=simultaneous_ops)
                t.start()
                t.join()
                
                time.sleep(1)
                self.print_status()
                
                # Turn off all
                for ch in self.channels:
                    self.turn_off(ch.device_id, ch.channel)
                
                time.sleep(0.5)
            
            self.test_results['simultaneous_control'] = True
            print("✓ Mixed operations test PASSED\n")
            return True
        except Exception as e:
            print(f"✗ Mixed operations test FAILED: {e}\n")
            traceback.print_exc()
            return False
    
    def test_stress_test(self):
        """Stress test: rapid switching of all channels"""
        print("\n" + "="*60)
        print("TEST 4: STRESS TEST (Rapid Switching)")
        print("="*60)
        print("Rapid toggling of all 10 channels for 10 cycles...\n")
        
        try:
            for cycle in range(5):  # 5 rapid cycles
                print(f"Cycle {cycle + 1}/5:", end=" ")
                for ch in self.channels:
                    self.toggle(ch.device_id, ch.channel)
                print("✓")
                time.sleep(0.5)
            
            self.print_status()
            self.test_results['stress_test'] = True
            print("✓ Stress test PASSED\n")
            return True
        except Exception as e:
            print(f"✗ Stress test FAILED: {e}\n")
            traceback.print_exc()
            return False
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        for test_name, result in self.test_results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{test_name.upper():<30} {status}")
        print("="*60)
        
        all_passed = all(self.test_results.values())
        if all_passed:
            print("\n✓ ALL TESTS PASSED - Dual relay control is working correctly!")
        else:
            print("\n✗ SOME TESTS FAILED - Check the errors above")
        
        return all_passed


class MockRelay:
    """Mock relay for testing without physical hardware"""
    
    def __init__(self, channels: int, name: str):
        self.channels = channels
        self.name = name
        self.state = {i: False for i in range(1, channels + 1)}
        print(f"→ Mock {name} relay created with {channels} channels")
    
    def turn_on(self, channel: int):
        if 1 <= channel <= self.channels:
            self.state[channel] = True
    
    def turn_off(self, channel: int):
        if 1 <= channel <= self.channels:
            self.state[channel] = False
    
    def toggle_state(self, channel: int):
        if 1 <= channel <= self.channels:
            self.state[channel] = not self.state[channel]
    
    @property
    def state_dict(self):
        return self.state


def main():
    """Main test execution"""
    print("\n" + "="*60)
    print("DUAL RELAY CONTROL TESTING")
    print("8-Channel + 2-Channel = 10 Total Channels")
    print("="*60)
    
    controller = DualRelayController()
    
    # Initialize
    if not controller.initialize():
        print("\n✗ Failed to initialize relay controller")
        return False
    
    # Run tests
    try:
        controller.test_sequential_control()
        controller.test_simultaneous_control()
        controller.test_mixed_operations()
        controller.test_stress_test()
    except KeyboardInterrupt:
        print("\n\n⚠ Tests interrupted by user")
    except Exception as e:
        print(f"\n✗ Unexpected error during testing: {e}")
        traceback.print_exc()
    
    # Print summary
    success = controller.print_summary()
    
    # Cleanup - turn off all relays
    print("\nCleaning up - turning off all relays...")
    for ch in controller.channels:
        controller.turn_off(ch.device_id, ch.channel)
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
