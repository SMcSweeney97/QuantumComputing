import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Plot():
    def __init__(self, history_1, history_2, history_3, history_4, history_5, history_6, epochs):
        for col in history_1.columns:
            plt.plot(np.arange(1, epochs + 1), history_1[col], label = 'Basis Encoding (0-1)')
            plt.plot(np.arange(1, epochs + 1), history_2[col], label = 'Basis Encoding (3-6)')
            plt.plot(np.arange(1, epochs + 1), history_3[col], label = 'Amplitude Encoding (0-1)')
            plt.plot(np.arange(1, epochs + 1), history_4[col], label = 'Amplitude Encoding (3-6)')
            plt.plot(np.arange(1, epochs + 1), history_5[col], label = 'FRQI (0-1)')
            plt.plot(np.arange(1, epochs + 1), history_6[col], label = 'FRQI (3-6)')
            
            plt.xlabel('Epochs')
            plt.ylabel(col)
            plt.legend()
            plt.xticks(np.arange(0, epochs + 1, 2))
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.savefig('D:/OneDrive/Desktop/{}.png'.format(col), bbox_inches = 'tight', pad_inches = 0)
            plt.show()
    
def main():
    history_1 = pd.read_csv('D:/OneDrive/Desktop/Image Encoding/QML Basis Encoding (0-1).csv')
    history_2 = pd.read_csv('D:/OneDrive/Desktop/Image Encoding/QML Basis Encoding (3-6).csv')
    history_3 = pd.read_csv('D:/OneDrive/Desktop/Image Encoding/Amplitude Encoding (0-1).csv')
    history_4 = pd.read_csv('D:/OneDrive/Desktop/Image Encoding/Amplitude Encoding (3-6).csv')
    history_5 = pd.read_csv('D:/OneDrive/Desktop/Image Encoding/FRQI (0-1).csv')
    history_6 = pd.read_csv('D:/OneDrive/Desktop/Image Encoding/FRQI (3-6).csv')
    Plot(history_1, history_2, history_3, history_4, history_5, history_6, epochs = 20)

if __name__ == '__main__':
    main()
