import numpy as np
import matplotlib.pyplot as plt

import scansegmentapi.compact as CompactApi
from scansegmentapi.udp_handler import UDPHandler
import keyboard

UDP_IP = "172.16.35.58"
UDP_PORT = 2122

transport = UDPHandler(UDP_IP, UDP_PORT, 65535)
receiver = CompactApi.Receiver(transport)

temp_frame = 0

segments = []
frame_numbers = []
segment_counters = []

while True:
    
    segment, frame_number, segment_counter = receiver.receive_segments(1)
    #print(segments[0])
    #print(f'Info segmenti: {segments[0]["Modules"][0]["NumberOfBeamsPerScan"]}')
    #print(f'Segment counters: {segment_counters}')
    #print(f'Frame numbers: {frame_numbers}')
    segments.append(segment[0])
    frame_numbers.append(frame_number[0])
    segment_counters.append(segment_counter[0])
    #print(frame_number[0])
    if len(frame_numbers) == 10 and len(segment_counters) == 10:#  & np.all(np.array(frame_numbers) == frame_numbers[0]) & np.all(segment_counters[:-1] < segment_counters[1:]):
        if np.all(np.array(frame_numbers) == frame_numbers[0]):
            if np.all(np.array(segment_counters[:-1]) < np.array(segment_counters[1:])):
                print("Ricevuti 10 segmenti di uno stesso frame in ordine corretto")
                print(f'Frame number: {frame_numbers}')
                print(f'Segment counters: {segment_counters}')
                frame_numbers = []
                segment_counters = []
                segments = []
                


        else:
            mask = np.array(frame_numbers) != frame_numbers[0]
            frame_numbers = list(np.array(frame_numbers)[mask])
            segment_counters = list(np.array(segment_counters)[mask])
            segments = list(np.array(segments)[mask])

        


    
    # if segments[1]["Modules"][0]["SegmentCounter"] != segment_counters[3]:
    #     print("Errore: Segment counter non corrispondente!")
    #print(f'Frame numbers: {frame_numbers}')
    #print(f'Segment counters: {segment_counters}')


    # if segments[0]["Modules"][0]["FrameNumber"] != temp_frame:
    #     temp_frame = segments[0]["Modules"][0]["FrameNumber"]
    #     print(segment_counters)
    #     print(f'Nuovo frame ricevuto: {temp_frame}')
    if keyboard.is_pressed('q'):
        break

receiver.close_connection()
