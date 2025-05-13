
import matplotlib.pyplot as plt

def graphing(diff_percentage, preds, targets):
    plt.subplot(2,1,1)
    plt.plot(preds)
    plt.plot(targets)
    plt.subplot(2,1,2)
    plt.plot(diff_percentage)
    
    #plt.plot(diff_percentage)
    plt.show()