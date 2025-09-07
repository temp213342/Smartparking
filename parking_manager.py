from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import uuid
import json
from dataclasses import dataclass, asdict
from enum import Enum

class SlotStatus(Enum):
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    RESERVED = "reserved"

@dataclass
class ParkingSlot:
    slot_id: str
    status: SlotStatus
    vehicle_type: Optional[str] = None
    vehicle_number: Optional[str] = None
    customer_name: Optional[str] = None
    arrival_time: Optional[datetime] = None
    pickup_time: Optional[datetime] = None
    reservation_time: Optional[datetime] = None

@dataclass
class Transaction:
    id: str
    slot_id: str
    vehicle_type: str
    vehicle_number: str
    customer_name: str
    arrival_time: datetime
    departure_time: datetime
    amount: float
    timestamp: datetime

class ParkingManager:
    def __init__(self, total_slots: int = 20):
        self.total_slots = total_slots
        self.slots: Dict[str, ParkingSlot] = {}
        self.transactions: List[Transaction] = []
        self.total_revenue: float = 0.0
        
        # Initialize slots
        for i in range(1, total_slots + 1):
            slot_id = f"slot_{i}"
            self.slots[slot_id] = ParkingSlot(
                slot_id=slot_id,
                status=SlotStatus.AVAILABLE
            )
        
        # Pricing configuration
        self.pricing = {
            'Car': {'base': 150, 'rush_surcharge': 30},
            'Bike': {'base': 200, 'rush_surcharge': 50}, 
            'Truck': {'base': 300, 'rush_surcharge': 70},
            'night_rate': 100
        }
        
        # Holiday data for 2025
        self.holidays_2025 = [
            {'date': '2025-01-01', 'holiday': 'New Year\'s Day'},
            {'date': '2025-01-26', 'holiday': 'Republic Day'},
            {'date': '2025-03-13', 'holiday': 'Holi'},
            {'date': '2025-03-31', 'holiday': 'Eid ul-Fitr'},
            {'date': '2025-04-14', 'holiday': 'Ram Navami'},
            {'date': '2025-04-18', 'holiday': 'Good Friday'},
            {'date': '2025-08-15', 'holiday': 'Independence Day'},
            {'date': '2025-08-27', 'holiday': 'Janmashtami'},
            {'date': '2025-10-02', 'holiday': 'Gandhi Jayanti'},
            {'date': '2025-10-22', 'holiday': 'Dussehra'},
            {'date': '2025-11-12', 'holiday': 'Diwali'},
            {'date': '2025-12-25', 'holiday': 'Christmas Day'}
        ]
    
    def is_rush_hour(self, dt: datetime) -> bool:
        """Check if datetime is during rush hours"""
        # Friday 5PM-12AM
        if dt.weekday() == 4 and dt.hour >= 17:
            return True
        
        # Weekend 11AM-12AM  
        if dt.weekday() in [5, 6] and dt.hour >= 11:
            return True
        
        # Check holidays
        date_str = dt.strftime('%Y-%m-%d')
        for holiday in self.holidays_2025:
            if holiday['date'] == date_str:
                return True
        
        return False
    
    def is_night_hour(self, dt: datetime) -> bool:
        """Check if datetime is during night hours (11PM-5AM)"""
        return dt.hour >= 23 or dt.hour < 5
    
    def calculate_parking_fee(self, vehicle_type: str, arrival_dt: datetime, departure_dt: datetime) -> Dict:
        """Calculate parking fee based on vehicle type and duration"""
        duration_hours = max(1, (departure_dt - arrival_dt).total_seconds() / 3600)
        duration_hours = round(duration_hours, 2)
        
        base_rate = self.pricing[vehicle_type]['base']
        rush_surcharge = self.pricing[vehicle_type]['rush_surcharge']
        
        total_cost = 0
        rush_hours = 0
        night_hours = 0
        regular_hours = 0
        
        current_dt = arrival_dt
        while current_dt < departure_dt:
            next_hour = current_dt + timedelta(hours=1)
            if next_hour > departure_dt:
                hour_fraction = (departure_dt - current_dt).total_seconds() / 3600
            else:
                hour_fraction = 1
            
            if self.is_night_hour(current_dt):
                night_hours += hour_fraction
                total_cost += self.pricing['night_rate'] * hour_fraction
            elif self.is_rush_hour(current_dt):
                rush_hours += hour_fraction
                total_cost += (base_rate + rush_surcharge) * hour_fraction
            else:
                regular_hours += hour_fraction
                total_cost += base_rate * hour_fraction
            
            current_dt = next_hour
        
        return {
            'duration_hours': duration_hours,
            'regular_hours': regular_hours,
            'rush_hours': rush_hours,
            'night_hours': night_hours,
            'base_rate': base_rate,
            'rush_surcharge': rush_surcharge,
            'night_rate': self.pricing['night_rate'],
            'total_cost': round(total_cost, 2)
        }
    
    def get_available_slots(self) -> List[str]:
        """Get list of available parking slots"""
        return [slot_id for slot_id, slot in self.slots.items() 
                if slot.status == SlotStatus.AVAILABLE]
    
    def get_occupied_slots(self) -> List[str]:
        """Get list of occupied parking slots"""
        return [slot_id for slot_id, slot in self.slots.items() 
                if slot.status == SlotStatus.OCCUPIED]
    
    def get_reserved_slots(self) -> List[str]:
        """Get list of reserved parking slots"""
        return [slot_id for slot_id, slot in self.slots.items() 
                if slot.status == SlotStatus.RESERVED]
    
    def park_vehicle(self, slot_id: str, vehicle_type: str, vehicle_number: str, 
                    customer_name: str, arrival_dt: datetime, pickup_dt: datetime) -> bool:
        """Park a vehicle in the specified slot"""
        if slot_id not in self.slots or self.slots[slot_id].status != SlotStatus.AVAILABLE:
            return False
        
        self.slots[slot_id] = ParkingSlot(
            slot_id=slot_id,
            status=SlotStatus.OCCUPIED,
            vehicle_type=vehicle_type,
            vehicle_number=vehicle_number.upper(),
            customer_name=customer_name,
            arrival_time=arrival_dt,
            pickup_time=pickup_dt
        )
        
        return True
    
    def reserve_slot(self, slot_id: str, vehicle_type: str, vehicle_number: str, 
                    customer_name: str, reservation_dt: datetime, duration_hours: int) -> bool:
        """Reserve a parking slot"""
        if slot_id not in self.slots or self.slots[slot_id].status != SlotStatus.AVAILABLE:
            return False
        
        pickup_dt = reservation_dt + timedelta(hours=duration_hours)
        
        self.slots[slot_id] = ParkingSlot(
            slot_id=slot_id,
            status=SlotStatus.RESERVED,
            vehicle_type=vehicle_type,
            vehicle_number=vehicle_number.upper(),
            customer_name=customer_name,
            pickup_time=pickup_dt,
            reservation_time=reservation_dt
        )
        
        return True
    
    def remove_vehicle(self, slot_id: str, departure_dt: datetime) -> Optional[Dict]:
        """Remove a vehicle and generate bill"""
        if slot_id not in self.slots or self.slots[slot_id].status != SlotStatus.OCCUPIED:
            return None
        
        slot = self.slots[slot_id]
        
        bill_data = self.calculate_parking_fee(
            slot.vehicle_type,
            slot.arrival_time,
            departure_dt
        )
        
        transaction = Transaction(
            id=str(uuid.uuid4()),
            slot_id=slot_id,
            vehicle_type=slot.vehicle_type,
            vehicle_number=slot.vehicle_number,
            customer_name=slot.customer_name,
            arrival_time=slot.arrival_time,
            departure_time=departure_dt,
            amount=bill_data['total_cost'],
            timestamp=datetime.now()
        )
        
        self.transactions.append(transaction)
        self.total_revenue += bill_data['total_cost']
        
        # Clear the slot
        self.slots[slot_id] = ParkingSlot(
            slot_id=slot_id,
            status=SlotStatus.AVAILABLE
        )
        
        return {**bill_data, **asdict(transaction)}
    
    def search_vehicle(self, query: str) -> List[Tuple[str, ParkingSlot]]:
        """Search for vehicle by number"""
        results = []
        for slot_id, slot in self.slots.items():
            if (slot.vehicle_number and 
                query.upper() in slot.vehicle_number.upper()):
                results.append((slot_id, slot))
        return results
    
    def get_slot_data(self, slot_id: str) -> Optional[ParkingSlot]:
        """Get slot data"""
        return self.slots.get(slot_id)
    
    def get_statistics(self) -> Dict:
        """Get parking statistics"""
        available_count = len(self.get_available_slots())
        occupied_count = len(self.get_occupied_slots())
        reserved_count = len(self.get_reserved_slots())
        occupancy_rate = (occupied_count / self.total_slots) * 100
        
        return {
            'total_slots': self.total_slots,
            'available_count': available_count,
            'occupied_count': occupied_count,
            'reserved_count': reserved_count,
            'occupancy_rate': round(occupancy_rate, 1),
            'total_revenue': self.total_revenue,
            'total_transactions': len(self.transactions)
        }
    
    def get_recent_transactions(self, limit: int = 10) -> List[Dict]:
        """Get recent transactions"""
        sorted_transactions = sorted(
            self.transactions,
            key=lambda x: x.departure_time,
            reverse=True
        )
        
        return [asdict(t) for t in sorted_transactions[:limit]]
    
    def clear_all_data(self):
        """Clear all parking data (admin function)"""
        for i in range(1, self.total_slots + 1):
            slot_id = f"slot_{i}"
            self.slots[slot_id] = ParkingSlot(
                slot_id=slot_id,
                status=SlotStatus.AVAILABLE
            )
        
        self.transactions = []
        self.total_revenue = 0.0

# Global instance
parking_manager = ParkingManager()
