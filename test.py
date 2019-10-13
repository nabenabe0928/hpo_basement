class parent():
    def __init__(self, a, b, c="ccc"):
        self.a = a
        self.b = b
        self.c = c

class child(parent):
    def __init__(self, a,b,d={}):
        super().__init__(a,b)
        print(self.a)

    def ppp(self):
        print(self.c)

aaa = child(a=1,b=1.2)
aaa.ppp()