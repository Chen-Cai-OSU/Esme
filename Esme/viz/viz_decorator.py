import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(data={'foo':list(range(5)), 'bar':list(range(5, 10, 1))})

def save_fig(**param):
    # https://stackoverflow.com/questions/53449782/using-a-decorator-to-save-matplotlib-graphs-saved-output-is-blank
    def outer(func):
        def inner(*args, **kwargs):
            artist = func(*args)
            if 'filename' in param.keys():
                print('filename: %s'%param['filename'])
                plt.savefig(param['filename'])
            if 'show' in param.keys() and param["show"]:
                print('show')
                plt.show()
            else:
                return artist
        return inner
    return outer

@save_fig(**{'filename': 'foo.png', 'show' : True})
def plot_this():
    return plt.scatter(df['foo'], df['bar'])


if __name__ == "__main__":
    plot_this()