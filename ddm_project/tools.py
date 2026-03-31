import matplotlib.pyplot as plt


def plot_res(res,filename,title=None):
    figure =plt.figure()
    plt.plot(res)
    plt.yscale("log")
    plt.ylabel("res")
    if title is not None:
        plt.title(title)
    plt.savefig(filename)
    plt.close()