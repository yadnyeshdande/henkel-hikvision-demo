"""
Production-ready dual relay controller for 10-channel system (8ch + 2ch).
Can work simultaneously, sequentially, or in mixed patterns.
"""

import threading
import time
from typing import List, Tuple, Callable, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class RelayMode(Enum):
    """Relay operation modes"""
    SEQUENTIAL = "sequential"
    SIMULTANEOUS = "simultaneous"
    MIXED = "mixed"


@dataclass
class RelayEvent:
    """Log entry for relay operations"""
    timestamp: datetime
    device_id: int
    channel: int
    action: str
    status: str


class DualRelaySystem:
    """
    10-channel relay control system: 8-channel device + 2-channel device.
    Supports sequential, simultaneous, and mixed control patterns.
    Thread-safe with lock mechanism for concurrent operations.
    """
    
    def __init__(self, auto_initialize: bool = True, enable_logging: bool = True):
        self.relay_8ch = None
        self.relay_2ch = None
        self.lock = threading.RLock()
        self.channels = {
            1: {i: False for i in range(1, 9)},      # 8-channel device
            2: {i: False for i in range(1, 3)}       # 2-channel device
        }
        self.enable_logging = enable_logging
        self.event_log: List[RelayEvent] = []
        self.callbacks: List[Callable] = []
        
        if auto_initialize:
            self.initialize()
    
    def initialize(self) -> bool:
        """Initialize both relay modules"""
        try:
            import pyhid_usb_relay
            
            # Try to find relay devices
            relays = pyhid_usb_relay.find_all() if hasattr(pyhid_usb_relay, 'find_all') else [pyhid_usb_relay.find()]
            
            if isinstance(relays, list):
                if len(relays) >= 1:
                    self.relay_8ch = relays[0]
                if len(relays) >= 2:
                    self.relay_2ch = relays[1]
            elif relays:
                self.relay_8ch = relays
            
            return self.relay_8ch is not None
        except ImportError:
            return False
        except Exception:
            return False
    
    def _log_event(self, device_id: int, channel: int, action: str, status: str = "OK"):
        """Log relay operation event"""
        if self.enable_logging:
            event = RelayEvent(
                timestamp=datetime.now(),
                device_id=device_id,
                channel=channel,
                action=action,
                status=status
            )
            self.event_log.append(event)
    
    def _notify_callbacks(self, device_id: int, channel: int, state: bool):
        """Notify registered callbacks of state changes"""
        for callback in self.callbacks:
            try:
                callback(device_id, channel, state)
            except Exception:
                pass
    
    def register_callback(self, callback: Callable):
        """Register a callback function for state change events"""
        self.callbacks.append(callback)
    
    def turn_on(self, device_id: int, channel: int) -> bool:
        """Turn on a single relay channel"""
        with self.lock:
            try:
                if device_id == 1 and channel in range(1, 9):
                    if self.relay_8ch:
                        self.relay_8ch.turn_on(channel)
                    self.channels[1][channel] = True
                    self._log_event(device_id, channel, "ON")
                    self._notify_callbacks(device_id, channel, True)
                    return True
                
                elif device_id == 2 and channel in range(1, 3):
                    if self.relay_2ch:
                        self.relay_2ch.turn_on(channel)
                    self.channels[2][channel] = True
                    self._log_event(device_id, channel, "ON")
                    self._notify_callbacks(device_id, channel, True)
                    return True
                
                return False
            except Exception as e:
                self._log_event(device_id, channel, "ON", f"ERROR: {str(e)}")
                return False
    
    def turn_off(self, device_id: int, channel: int) -> bool:
        """Turn off a single relay channel"""
        with self.lock:
            try:
                if device_id == 1 and channel in range(1, 9):
                    if self.relay_8ch:
                        self.relay_8ch.turn_off(channel)
                    self.channels[1][channel] = False
                    self._log_event(device_id, channel, "OFF")
                    self._notify_callbacks(device_id, channel, False)
                    return True
                
                elif device_id == 2 and channel in range(1, 3):
                    if self.relay_2ch:
                        self.relay_2ch.turn_off(channel)
                    self.channels[2][channel] = False
                    self._log_event(device_id, channel, "OFF")
                    self._notify_callbacks(device_id, channel, False)
                    return True
                
                return False
            except Exception as e:
                self._log_event(device_id, channel, "OFF", f"ERROR: {str(e)}")
                return False
    
    def toggle(self, device_id: int, channel: int) -> bool:
        """Toggle a relay channel"""
        current_state = self.get_state(device_id, channel)
        if current_state:
            return self.turn_off(device_id, channel)
        else:
            return self.turn_on(device_id, channel)
    
    def get_state(self, device_id: int, channel: int) -> Optional[bool]:
        """Get current state of a relay channel"""
        with self.lock:
            if device_id in self.channels and channel in self.channels[device_id]:
                return self.channels[device_id][channel]
        return None
    
    def turn_on_sequential(self, channels: List[Tuple[int, int]], delay: float = 0.1) -> bool:
        """
        Turn on multiple channels sequentially with delay between each.
        Args:
            channels: List of (device_id, channel) tuples
            delay: Delay in seconds between operations
        """
        try:
            for device_id, channel in channels:
                if not self.turn_on(device_id, channel):
                    return False
                time.sleep(delay)
            return True
        except Exception:
            return False
    
    def turn_off_sequential(self, channels: List[Tuple[int, int]], delay: float = 0.1) -> bool:
        """Turn off multiple channels sequentially with delay between each."""
        try:
            for device_id, channel in channels:
                if not self.turn_off(device_id, channel):
                    return False
                time.sleep(delay)
            return True
        except Exception:
            return False
    
    def turn_on_simultaneous(self, channels: List[Tuple[int, int]]) -> bool:
        """Turn on multiple channels simultaneously using threading."""
        threads = []
        results = {}
        
        def turn_on_channel(device_id, channel):
            results[(device_id, channel)] = self.turn_on(device_id, channel)
        
        try:
            for device_id, channel in channels:
                t = threading.Thread(target=turn_on_channel, args=(device_id, channel))
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            return all(results.values()) if results else False
        except Exception:
            return False
    
    def turn_off_simultaneous(self, channels: List[Tuple[int, int]]) -> bool:
        """Turn off multiple channels simultaneously using threading."""
        threads = []
        results = {}
        
        def turn_off_channel(device_id, channel):
            results[(device_id, channel)] = self.turn_off(device_id, channel)
        
        try:
            for device_id, channel in channels:
                t = threading.Thread(target=turn_off_channel, args=(device_id, channel))
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            return all(results.values()) if results else False
        except Exception:
            return False
    
    def turn_all_on(self, mode: RelayMode = RelayMode.SEQUENTIAL, delay: float = 0.1) -> bool:
        """Turn on all 10 relay channels"""
        all_channels = [(1, i) for i in range(1, 9)] + [(2, i) for i in range(1, 3)]
        
        if mode == RelayMode.SEQUENTIAL:
            return self.turn_on_sequential(all_channels, delay)
        elif mode == RelayMode.SIMULTANEOUS:
            return self.turn_on_simultaneous(all_channels)
        return False
    
    def turn_all_off(self, mode: RelayMode = RelayMode.SEQUENTIAL, delay: float = 0.1) -> bool:
        """Turn off all 10 relay channels"""
        all_channels = [(1, i) for i in range(1, 9)] + [(2, i) for i in range(1, 3)]
        
        if mode == RelayMode.SEQUENTIAL:
            return self.turn_off_sequential(all_channels, delay)
        elif mode == RelayMode.SIMULTANEOUS:
            return self.turn_off_simultaneous(all_channels)
        return False
    
    def get_all_states(self) -> dict:
        """Get state of all channels"""
        with self.lock:
            return {
                "8-channel": self.channels[1].copy(),
                "2-channel": self.channels[2].copy()
            }
    
    def print_status(self):
        """Print current status of all channels"""
        states = self.get_all_states()
        print("\n" + "="*50)
        print("RELAY STATUS - 10 Channels Total")
        print("="*50)
        print(f"{'Device':<15} {'Channel':<10} {'State':<10}")
        print("-"*50)
        
        for i in range(1, 9):
            state = "ON" if states["8-channel"][i] else "OFF"
            print(f"{'8-channel':<15} {i:<10} {state:<10}")
        
        print("-"*50)
        for i in range(1, 3):
            state = "ON" if states["2-channel"][i] else "OFF"
            print(f"{'2-channel':<15} {i:<10} {state:<10}")
        
        print("="*50 + "\n")
    
    def get_event_log(self) -> List[RelayEvent]:
        """Get list of all logged events"""
        return self.event_log.copy()
    
    def clear_event_log(self):
        """Clear the event log"""
        self.event_log.clear()
    
    def emergency_stop(self):
        """Emergency stop: turn off all relays immediately"""
        with self.lock:
            for ch in range(1, 9):
                self.turn_off(1, ch)
            for ch in range(1, 3):
                self.turn_off(2, ch)


# Example usage
if __name__ == "__main__":
    # Create system
    relay_sys = DualRelaySystem()
    
    # Turn on all channels sequentially
    print("Turning ON all channels (sequential)...")
    relay_sys.turn_all_on(RelayMode.SEQUENTIAL, delay=0.1)
    relay_sys.print_status()
    
    time.sleep(1)
    
    # Turn off all channels simultaneously
    print("Turning OFF all channels (simultaneous)...")
    relay_sys.turn_all_off(RelayMode.SIMULTANEOUS)
    relay_sys.print_status()
    
    # Mixed example: Turn on 8-channel sequentially, 2-channel simultaneously
    print("Mixed mode: 8-channel sequential + 2-channel simultaneous...")
    channels_8ch = [(1, i) for i in range(1, 9)]
    channels_2ch = [(2, i) for i in range(1, 3)]
    
    relay_sys.turn_on_sequential(channels_8ch, delay=0.1)
    relay_sys.turn_on_simultaneous(channels_2ch)
    relay_sys.print_status()
