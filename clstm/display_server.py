import os
import numpy
from pylab import *
import traceback
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
addr = os.environ.get("PYSERVER","tcp://127.0.0.1:9876")
socket.bind(addr)
poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)

def farg(index):
    global args
    return numpy.fromstring(args[index],dtype=float32)
def farg2(index,d0,d1):
    global args
    return numpy.fromstring(args[index],dtype=float32).reshape(d0,d1)

while True:
    while True:
        evts = poller.poll(100)
        if evts!=[]: break
        ginput(1,0.01)
    args = socket.recv_multipart()
    print "----------------"
    print args[0]
    result = None
    try:
        exec args[0]
    except Exception,e:
        print "FAILED"
        traceback.print_exc()
    draw()
    socket.send(str(result))
