"""
DISCLAIMER:
This code is provided as an example only. It is intended for educational purposes and should not be used in production environments.
The author assumes no responsibility or liability for any errors or omissions in the content of this code. Use at your own risk.

This Python script sends a UDP broadcast message to discover a picoScan100 device on the network.
It constructs a specific hex sequence and sends it as a broadcast message.
The script then listens for responses from devices on the network and extracts relevant information from the responses,
including the IP address, subnet mask, default gateway, DHCP status, and MAC address.
"""

import socket
import random
import time

# Function to send a UDP broadcast message and listen for responses
def send_udp_broadcast(hex_sequence, broadcast_ip, port, interface_ip):
    try:
        # Create a UDP socket
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Allow sending of broadcast messages
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        # Bind the socket to the specified network interface
        udp_socket.bind((interface_ip, 0))
        
        # Convert the hex sequence to bytes
        message = bytes(hex_sequence)
        
        # Send the message to the broadcast address and specified port
        udp_socket.sendto(message, (broadcast_ip, port))
        
        print(f"UDP broadcast sent from {interface_ip} to {broadcast_ip}:{port}")
        
        # Listen for responses
        udp_socket.settimeout(5)  # Set a timeout for receiving responses
        while True:
            try:
                data, addr = udp_socket.recvfrom(1024)  # Buffer size is 1024 bytes
                
                # Extract information based on identifiers with a two-byte shift
                ip_address = extract_info(data, b'EIPa', 4, shift=2)        # extracts the IP address
                subnet_mask = extract_info(data, b'ENMa', 4, shift=2)       # extracts the subnet mask
                default_gateway = extract_info(data, b'EDGa', 4, shift=2)   # extracts the default gateway
                dhcp_status = extract_info(data, b'EDhc', 1, shift=2)       # extracts the DHCP status
                mac_address = extract_info(data, b'EMAC', 6, shift=2)       # extracts the MAC address
                
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                
                print(f"Response at {timestamp} | IP: {ip_address} | Subnet: {subnet_mask} | Gateway: {default_gateway} | DHCP: {'Enabled' if dhcp_status == '01' else 'Disabled'} | MAC: {mac_address}")
            except socket.timeout:
                print("No more responses.")
                break
    
    except Exception as e:
        print(f"Error sending UDP broadcast: {e}")
    
    finally:
        # Close the socket
        udp_socket.close()

# Helper function to extract information from the response data
def extract_info(data, identifier, length, shift=0):
    index = data.find(identifier)
    if index != -1:
        value = data[index+len(identifier)+shift:index+len(identifier)+shift+length]
        if length == 4:
            return socket.inet_ntoa(value)
        elif length == 6:
            return ':'.join(f'{b:02x}' for b in value)
        elif length == 1:
            return f'{value[0]:02x}'
    return 'N/A'

# Helper function to generate a random byte list
def generate_random_byte_list(length):
    return [random.getrandbits(8) for _ in range(length)]

# Main function
if __name__ == "__main__":
    # Generate a random 4-byte Request ID
    requestID_hex = generate_random_byte_list(4)
    
    # Define the network interface IP
    interface_ip = '172.16.35.58'  # Replace with your network interface IP
    subnet_mask = '255.255.0.0' # Replace with your subnet mask
    
    # Split the IP address and subnet mask into bytes
    bytes_ip = interface_ip.split('.')
    bytes_subnet = subnet_mask.split('.')

    # Convert IP address and subnet mask to hex array
    bytes_ip = [int(b) for b in bytes_ip]
    bytes_subnet = [int(b) for b in bytes_subnet]

    # Define the hex sequence
    hex_sequence = [
        0x10,  # CMD
        0x00,  # isBC indicates whether telegram was sent as UDP broadcast or UDP unicast telegram (0x00 means Unicast, 0x01 means Broadcast)
        0x00, 0x08,  # length
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  # MAC Address of the host. For broadcasts the MAC Address is set to 0xFFFFFFFFFFFF
        requestID_hex[0], requestID_hex[1], requestID_hex[2], requestID_hex[3],  # Random Request ID: 32-bit random number for the RequestID for the outgoing telegram.
        0x01,  # Indicates that telegram is a scan telegram
        0x00,  # for future use
        bytes_ip[0], bytes_ip[1], bytes_ip[2], bytes_ip[3],  # IP address of interface 
        bytes_subnet[0], bytes_subnet[1], bytes_subnet[2], bytes_subnet[3]  # Subnet mask of interface 
    ]
    
    # Define the broadcast IP and the destination port
    broadcast_ip = '169.254.66.77'
    port = 2122
    
    # Send the UDP broadcast
    send_udp_broadcast(hex_sequence, broadcast_ip, port, interface_ip)