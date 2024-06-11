import matplotlib.pyplot as plt
import numpy as np

def visualize_predicted_attack(attr, pred_val):
    for j in range(7, 8):
        fig, ax = plt.subplots()  # Create a figure and an axes
        
        ax.plot(np.array(attr).T[j])
        ax.set_title('Atak przewidziany przez model')

        for i in range(len(pred_val)):
            if pred_val[i] > 0.5:
                ax.plot(i, attr[i][j], 'ro')
            else:
                ax.plot(i, attr[i][j], 'bo')

        plt.show()


def visualize_predicted_attack_multi_channel(attr, pred_val):
    for j in range(7, 8):
        fig, ax = plt.subplots()  # Create a figure and an axes
        
        ax.plot(np.array(attr).T[j])
        ax.set_title('Atak przewidziany przez model')

        for i in range(len(pred_val)):
            if pred_val[i] > 0.5:
                ax.plot(i, attr[i][j], 'ro')
            else:
                ax.plot(i, attr[i][j], 'bo')

        plt.show()