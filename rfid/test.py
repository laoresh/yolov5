import rfid as rf
import time
port = rf.L_port_init()
id_head = -1
id = list()
while True:
    iid = rf.read_id(port)
    print(iid)
    if iid != id_head and iid != -1:
        id_head = iid
        id.append(iid)
    print(id)
    port.flushInput()
    # time.sleep(0.01)

