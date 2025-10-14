
import matplotlib.pyplot as plt

def graphing(diff_percentage, preds, targets):
    timeframe = len(preds)
    plt.subplot(2,1,1).set_title("Prediction Vs Actual Price")
    plt.plot(preds, label="Prediction")
    plt.plot(targets[:timeframe], label="Actual")
    plt.legend()
    plt.subplot(2,1,2).set_title("Percentage difference")
    
    plt.plot(diff_percentage)
    plt.legend()

    #plt.plot(diff_percentage)
    plt.show()
