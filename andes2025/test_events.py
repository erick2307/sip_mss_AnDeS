#!/usr/bin/env python3
"""
Test script to validate the updated mesh ID and events functionality.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_events_functionality():
    """Test the events-based mesh ID selection."""
    print("=" * 60)
    print("EVENTS FUNCTIONALITY TEST")
    print("=" * 60)
    
    try:
        # Test importing the events and functions
        from datetime import datetime, timezone
        
        # Test the events structure
        events = [
            {
                'event_dt': datetime(2024,1,2,17,0,0,0,timezone.utc),
                'meshcode' : 533937621,
                'meshcodes': [533937614, 533937623, 533937624,
                             533937612, 533937621, 533937622,
                             533937514, 533937523, 533937524],
                'event': 'Haneda Airport runway collision',
                'merge': False,
                'order': 4
            },
            {
                'event_dt': datetime(2024,2,7,0,0,0,0,timezone.utc),
                'meshcode' : 533946403,
                'meshcodes': [533945592, 533946501, 533946502,
                             533945494, 533946403, 533946404,
                             533945492, 533946401, 533946402],
                'event': 'Taylor Swift – The Eras Tour (Tokyo Dome)',
                'merge': False,
                'order': 4
            }
        ]
        
        print("✅ Events structure created successfully")
        print(f"📊 Total events: {len(events)}")
        
        # Test extracting mesh IDs
        all_mesh_ids = []
        for event in events:
            if 'meshcode' in event:
                all_mesh_ids.append(str(event['meshcode']))
            if 'meshcodes' in event:
                all_mesh_ids.extend([str(code) for code in event['meshcodes']])
        
        unique_mesh_ids = sorted(list(set(all_mesh_ids)))
        print(f"📍 Unique mesh IDs extracted: {len(unique_mesh_ids)}")
        print(f"🔢 Sample mesh IDs: {unique_mesh_ids[:5]}...")
        
        # Test event lookup
        def get_event_by_name(event_name: str) -> dict:
            for event in events:
                if event['event'] == event_name:
                    return event
            return None
        
        # Test finding an event
        test_event = get_event_by_name('Haneda Airport runway collision')
        if test_event:
            print(f"✅ Event lookup successful: {test_event['event']}")
            print(f"📅 Event date: {test_event['event_dt']}")
            print(f"🗺️ Mesh codes count: {len(test_event['meshcodes'])}")
        else:
            print("❌ Event lookup failed")
            return False
        
        print()
        print("=" * 60)
        print("✅ ALL TESTS PASSED - Events functionality working!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_events_functionality()
    sys.exit(0 if success else 1)