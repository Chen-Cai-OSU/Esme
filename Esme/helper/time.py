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

import signal

def signal_handler(signum, frame):
    raise Exception("Timed out!")

def long_function_call(t = 1.5):
    time.sleep(t)

if __name__ == '__main__':
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(2)   # Ten seconds
    long_function_call(t=3)
    try:
        long_function_call(t=3.5)
    except:
        print ("Timed out!")
