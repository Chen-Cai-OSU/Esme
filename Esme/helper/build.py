import os

def rebuild():
    # todo undone
    dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(dir, '..','..')
    command = 'cd ' + file + ' && ./build.sh'
    print(command)
    os.system(command)