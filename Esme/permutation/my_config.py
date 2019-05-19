from sacred import Experiment
ex = Experiment()

@ex.config
def my_config():
    recipent = 'world'
    message = 'Hello %s' % recipent

@ex.automain
def my_main(message):
    print(message)

def config():
    a = [1, 2]
    b = [3, 4]
    from itertools import product
    list(product(a,b))

# if __name__ == '__main__':
#     ex.run_commandline()