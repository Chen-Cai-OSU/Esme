global a
a = 10

def f1():
    print f2()
    return f2()

def f2():
    return a

f1()