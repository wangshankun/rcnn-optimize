import ctypes
import time 
import sys
import random

data_size = (4*1024)                
seed       = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+=-/?><,.[]{};:~|"
send_data  = []
for i in range(data_size):
    send_data.append(random.choice(seed))

sendbuf  = (ctypes.c_char * data_size)(*send_data)


pdll = ctypes.cdll.LoadLibrary('./hardrock.so')
pdll.hardrock_init()

recvibuf = (ctypes.c_char * data_size)()
pdll.hardrock_write_fpga(sendbuf, 0, data_size)
pdll.hardrock_read_fpga(recvibuf, 0, data_size)

print cmp(sendbuf.value, recvibuf.value)

pdll.hardrock_exit()

