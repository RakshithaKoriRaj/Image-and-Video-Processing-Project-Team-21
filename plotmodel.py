import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

def create_acc_loss_graph(model_name):
    contents = open("model.log", "r").read().split("\n")

    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss = c.split(",")

            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))

            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))


    fig = plt.figure(dpi=500)
    #fig.tight_layout()
    plt.subplots_adjust(left  = 0.125, right = 0.9, bottom = 0.1, top = 1.0, wspace = 0.2, hspace = 0.3 )
    ax1 = plt.subplot2grid((4,1), (0,0))
    ax2 = plt.subplot2grid((4,1), (1,0), sharex=ax1)
    ax3 = plt.subplot2grid((4,1), (2,0), sharex=ax1)
    ax4 = plt.subplot2grid((4,1), (3,0), sharex=ax1)

    ax1.plot(range(len(times)), accuracies, label="acc")
    ax2.plot(range(len(times)), val_accs, label="val_acc")
    ax1.legend(loc=2)
    ax2.legend(loc=2)
    ax3.plot(range(len(times)),losses, label="loss")
    ax4.plot(range(len(times)),val_losses, label="val_loss")
    ax3.legend(loc=2)
    ax4.legend(loc=2)
    plt.savefig("{}.png".format(model_name), bbox_inches='tight')
    plt.show()

