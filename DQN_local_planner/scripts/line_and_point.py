## noktalar (0,0) ve (x, 0) ise ?? eğim sonsuz veya (0, y) ise eğim 0 oluyor??

from math import sqrt

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Line:
    def __init__(self, a=0, b=0, c=0):
        self.a = a
        self.b = b
        self.c = c

    def fromPoints(self, p1, p2):    
        self.a = p2.y - p1.y
        self.b = p1.x - p2.x
        self.c = p1.y*p2.x - p1.x*p2.y

    def fromSlopePoint(self, m, p):
        self.a = m
        self.b = -1
        self.c = p.y - m*p.x
        
    def closed_dist_with_sign(self, p):
        return (self.a * p.x + self.b * p.y + self.c) / sqrt(self.a**2 + self.b**2)
        
    def getSlope(self):
        return -self.a/self.b
        
    def print(self):
        print("A:{} B:{} C:{}".format(l1.a,l1.b,l1.c))
    
p1 = Point(0,0)
p2 = Point(3, 0)
robot = Point(3, 5)

l1 = Line()
l1.fromPoints(p1, p2)

# perpendicular line
l2 = Line()
m = -1.0 / l1.getSlope()
l2.fromSlopePoint(m, p1)
    
# dik doğrunun noktaya en yakın uzaklığı
#AP = m
#BP = -1
#CP = y1 - mx1

l1.print()

print("distance: {}".format(l1.closed_dist_with_sign(robot)))

print("distance: {}".format(l2.closed_dist_with_sign(robot)))