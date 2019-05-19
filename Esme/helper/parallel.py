from joblib import Parallel, delayed
import time

def unwrap_self(arg, **kwarg):
    return square_class.square_int(*arg, **kwarg)


class square_class:
    def square_int(self, i):
        time.sleep(1)
        return i * i

    def run(self, num):
        results = []
        results = Parallel(n_jobs=-1) \
            (delayed(unwrap_self)(i) for i in zip([self] * len(num), num))
        print(results)

