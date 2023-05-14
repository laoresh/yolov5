import time
import serial


def transfordata(s):
    up_16 = (s[0] // 16) * 10 + s[0] % 16
    low_16 = (s[1] // 16) * 10 + s[1] % 16
    id = up_16 * 100 + low_16
    return id


def L_port_init():
    try:
        serial_port = serial.Serial(
            port='/dev/ttyUSB0',
            baudrate=115200,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=0.005,
        )
        # Wait a second to let the port initialize
        time.sleep(1)
        print(serial_port.name)
        if not serial_port.is_open:
            serial_port.open()
        print("USB0 已打开")
        return serial_port
    except Exception as e:
        print("--异常————：", e)


def R_port_init():
    serial_port = serial.Serial(
        port='/dev/ttyUSB1',
        baudrate=115200,
        bytesize=8,
        parity='N',
        stopbits=1,
        timeout=0.005,
    )
    # Wait a second to let the port initialize
    time.sleep(1)
    print(serial_port.name)
    serial_port.close()
    serial_port.open()
    print("USB0 已打开")
    return serial_port


def read_id(serial_port):
    if serial_port.inWaiting:
        i = 7
        while serial_port.read() != b'\x07':
            i = i - 1
            if i < 0:
                return -1
    else:
        return -1
    if serial_port.read(3) == b'\x00\xee\x00':
        s = serial_port.read(2)
        id = transfordata(s)
        return id
    else:
        return -1

# port='/dev/ttyUSB1',baudrate=9600,bytesize=8,parity='N',stopbits=1,timeout=0.001
