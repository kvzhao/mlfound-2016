import math

def getU(x):
    u = x[0]
    v = x[1]
    return math.exp(u) + v*math.exp(u*v) + 2*u - 2*v - 3

def getV(x):
    u = x[0]
    v = x[1]
    return 2*math.exp(2*v) + u*math.exp(u*v) - 2*u + 4*v - 2

def getG(x):
    u = x[0]
    v = x[1]
    return (getU(x),getV(x))

def getH(x):
    u = x[0]
    v = x[1]
    return (math.exp(u)+v*v*math.exp(u*v)+2)*(4*math.exp(2*v)+u*u*math.exp(u*v)+4) - 2*(math.exp(u*v)+u*math.exp(u*v)-2)

x = [0,0]
for i in range(5):
    p = x
    q = getG(x)
    h = getH(x)
    k = ( p[0] - q[0]/h , p[1] - q[1]/h  )
    x = k
    print ('step %d' % (i+1))
    print (x)
