import time

def timefunction(method, threshold=1):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            if (te-ts)>threshold:
                print('%s takes %2.2f s' % (method.__name__, (te - ts) ))
        return result
    return timed

@timefunction
def h():
    time.sleep(1.1)
