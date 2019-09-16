class Foo:
    def __call__(self, a, b, c):
        return a+b+c

x = Foo()(1,2,3)
x(1, 2, 3) # __call__