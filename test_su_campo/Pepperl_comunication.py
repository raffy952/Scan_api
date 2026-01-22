import asyncio
import httpx
import struct
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

# --- Classi di Supporto ---

class StateConnection(Enum):
    DISCONNECTED = 0
    IN_CONNECTION = 1
    CONNECTED = 2
    ERROR = 3

@dataclass
class StateDev:
    state_connect: StateConnection = StateConnection.DISCONNECTED
    num_frame: int = 0

@dataclass
class DataNav310:
    microseconds: int = 0
    num_scan: int = 0
    scan_frequency: int = 0
    start_angle: int = 0
    resolution: int = 0
    dist: List[int] = field(default_factory=list)
    rssi: List[int] = field(default_factory=list)

@dataclass
class SensorConfig:
    scan_frequency: int = 50
    scan_direction: str = "ccw"
    samples_per_scan: int = 3600
    watchdog: str = "on"
    watchdog_timeout: int = 2000
    packet_type: str = "B"
    packet_crc: str = "none"
    start_angle: int = 0
    max_num_points_scan: int = 0
    skip_scans: int = 0

class PacketResult(Enum):
    IN_PROGRESS = 0
    SCAN_COMPLETE = 1
    WRONG_PACKET = 2
    INSUFFICIENT_DATA = 3

# --- Client Principale ---

class R2000TcpClient:
    HEADER_SIZE = 8
    MAGIC_NUMBER = b'\x5c\xa2' # 0xa25c in Little Endian

    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=5.0)
        self.expected_packet_number = 1
        self.packet_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    async def start_async(self, state_report: StateDev):
        while not self._stop_event.is_set():
            try:
                state_report.state_connect = StateConnection.IN_CONNECTION
                state_report.num_frame = 0
                
                await self.connect_and_receive_data(state_report)
            
            except asyncio.CancelledError:
                print("Operazione annullata")
                break
            except Exception as e:
                print(f"Errore nel loop principale: {e}")
                state_report.state_connect = StateConnection.ERROR
                if not self._stop_event.is_set():
                    await asyncio.sleep(5)

    async def connect_and_receive_data(self, state_report: StateDev):
        sensor_ip = "169.254.62.51"
        config = SensorConfig()

        # Configurazione via HTTP
        await self.configure_sensor(sensor_ip, config)
        port, handle = await self.request_tcp_handle(sensor_ip, config)
        print(f"Porta: {port}, Handle: {handle}")

        reader, writer = None, None
        try:
            # Connessione TCP
            reader, writer = await asyncio.open_connection(sensor_ip, port)
            
            # Avvio scansione
            await self.start_scan_output(sensor_ip, handle)
            state_report.state_connect = StateConnection.CONNECTED

            # Ricezione dati
            await self.receive_scan_data(reader, writer, state_report)
            
        finally:
            if writer:
                try:
                    await self.stop_scan_output(sensor_ip, handle)
                    await self.release_handle(sensor_ip, handle)
                    writer.close()
                    await writer.wait_closed()
                except Exception as e:
                    print(f"Errore durante cleanup: {e}")

    async def configure_sensor(self, ip, config):
        url = (f"http://{ip}/cmd/set_parameter?"
               f"scan_frequency={config.scan_frequency}&"
               f"scan_direction={config.scan_direction}&"
               f"samples_per_scan={config.samples_per_scan}")
        await self.http_client.get(url)

    async def request_tcp_handle(self, ip, config):
        url = (f"http://{ip}/cmd/request_handle_tcp?"
               f"packet_type={config.packet_type}&"
               f"packet_crc={config.packet_crc}&"
               f"watchdog={config.watchdog}&"
               f"watchdogtimeout={config.watchdog_timeout}&"
               f"start_angle={config.start_angle}&"
               f"max_num_points_scan={config.max_num_points_scan}&"
               f"skip_scans={config.skip_scans}")
        
        resp = await self.http_client.get(url)
        data = resp.json() # Il sensore solitamente risponde in JSON
        return data['port'], data['handle']

    async def start_scan_output(self, ip, handle):
        await self.http_client.get(f"http://{ip}/cmd/start_scanoutput?handle={handle}")

    async def stop_scan_output(self, ip, handle):
        await self.http_client.get(f"http://{ip}/cmd/stop_scanoutput?handle={handle}")

    async def release_handle(self, ip, handle):
        await self.http_client.get(f"http://{ip}/cmd/release_handle?handle={handle}")

    async def receive_scan_data(self, reader, writer, state_report):
        feed_watchdog = b'\x66\x65\x65\x64\x77\x64\x67\x04'
        
        first_packet_info = {}
        distance_list = []
        amplitude_list = []

        async with self.packet_lock:
            self.expected_packet_number = 1

        while not self._stop_event.is_set():
            try:
                # 1. Leggi l'header fisso di 8 byte
                header_data = await reader.readexactly(self.HEADER_SIZE)
                
                # 2. Verifica Magic Number (0xa25c in Little Endian)
                if header_data[0:2] != self.MAGIC_NUMBER:
                    # Se non sincronizzato, leggiamo un byte alla volta finché non lo troviamo
                    continue

                # 3. Leggi la dimensione totale del pacchetto (offset 4, 4 byte)
                packet_size = struct.unpack("<I", header_data[4:8])[0]
                
                if packet_size < self.HEADER_SIZE or packet_size > 100000:
                    print(f"Dimensione pacchetto non valida: {packet_size}")
                    continue

                # 4. Leggi il corpo rimanente
                remaining_bytes = packet_size - self.HEADER_SIZE
                payload = await reader.readexactly(remaining_bytes)
                full_packet = header_data + payload

                # 5. Feed watchdog (scriviamo al sensore per dirgli che siamo vivi)
                writer.write(feed_watchdog)
                await writer.drain()

                # 6. Processa il pacchetto completo
                result = await self.process_packet(full_packet, first_packet_info, distance_list, amplitude_list)

                if result == PacketResult.SCAN_COMPLETE:
                    state_report.num_frame += 1
                    
            except asyncio.IncompleteReadError:
                print("Connessione chiusa dal sensore.")
                break
            except Exception as e:
                print(f"Errore durante la ricezione: {e}")
                raise

    async def process_packet(self, data, first_packet, distance_list, amplitude_list):
        # Header pacchetto R2000:
        # Offset 10: scan_number (H = unsigned short)
        # Offset 12: packet_number (H)
        # Offset 34: scan_frequency (I = unsigned int)
        # Offset 38: num_points_scan (H)
        # Offset 40: num_points_packet (H)
        # Offset 44: first_angle (i = signed int)
        # Offset 48: angular_increment (i = signed int)
        
        scan_number = struct.unpack_from("<H", data, 10)[0]
        packet_number = struct.unpack_from("<H", data, 12)[0]
        num_points_scan = struct.unpack_from("<H", data, 38)[0]
        num_points_packet = struct.unpack_from("<H", data, 40)[0]

        async with self.packet_lock:
            if packet_number != self.expected_packet_number:
                print(f"Pacchetto fuori sequenza: ricevuto {packet_number}, atteso {self.expected_packet_number}")
                self.expected_packet_number = 1
                distance_list.clear()
                amplitude_list.clear()
                return PacketResult.WRONG_PACKET
            self.expected_packet_number += 1

        if packet_number == 1:
            first_packet['micro'] = int(time.time() * 1000000) % 0xFFFFFFFF
            first_packet['freq'] = struct.unpack_from("<I", data, 34)[0]
            first_packet['angle'] = struct.unpack_from("<i", data, 44)[0]
            first_packet['increment'] = struct.unpack_from("<i", data, 48)[0]

        # I dati dei punti iniziano solitamente all'offset 76
        # Ogni punto è 6 byte: 4 byte distanza (I) + 2 byte RSSI (H)
        offset = 76
        for _ in range(num_points_packet):
            if offset + 6 > len(data): break
            
            dist = struct.unpack_from("<I", data, offset)[0]
            if dist == 0xFFFFFFFF: dist = 60000 # Valore di errore/infinito
            
            rssi = struct.unpack_from("<H", data, offset + 4)[0]
            
            distance_list.append(dist)
            amplitude_list.append(rssi << 4) # Scalato come nel codice C#
            offset += 6

        # Se abbiamo accumulato tutti i punti della scansione
        if len(distance_list) >= num_points_scan:
            scan_data = DataNav310(
                microseconds=first_packet.get('micro', 0),
                num_scan=num_points_scan,
                scan_frequency=first_packet.get('freq', 0) // 1000,
                start_angle=first_packet.get('angle', 0) // 10000,
                resolution=first_packet.get('increment', 0),
                dist=list(distance_list),
                rssi=list(amplitude_list)
            )
            
            self.print_basic_stats(scan_data)
            
            # Reset per la prossima scansione completa
            distance_list.clear()
            amplitude_list.clear()
            async with self.packet_lock:
                self.expected_packet_number = 1
            return PacketResult.SCAN_COMPLETE

        return PacketResult.IN_PROGRESS
    def print_basic_stats(self, data: DataNav310):
        print(f"\n--- Scansione Ricevuta ---")
        print(f"Punti: {data.num_scan} | Freq: {data.scan_frequency}Hz")
        valid = [d for d in data.dist if d < 60000]
        if valid:
            print(f"Distanza Min: {min(valid)}mm | Max: {max(valid)}mm")

    def stop(self):
        self._stop_event.set()

# --- Esempio di utilizzo ---

async def main():
    state = StateDev()
    client = R2000TcpClient()
    
    try:
        await client.start_async(state)
    except KeyboardInterrupt:
        client.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass