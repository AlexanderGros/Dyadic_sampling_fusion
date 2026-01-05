# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:55:38 2024

@author: Alexander

combines snr vs accuracy curve of duadic method and classical IQ on same graph for comparison purposes
"""

#%%
#imports

import numpy as np
import matplotlib.pyplot as plt



#%%
# !!!
# Plot accuracy curve

snr_hist = [-1.5, 1.5, 4.5, 9]
acc_hist_dy =[0.6885245901639344, 0.9211214953271029, 0.948976049982645, 0.950121800247594]
acc_hist_iq = [0.6229508196721312, 0.845981308411215, 0.8660187434918432, 0.8741492489227176]


# orig 20 fontsize and 6 for linewidth
plt.figure()
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)
plt.plot(snr_hist, acc_hist_dy, linewidth=9, label='Dyadic scheme' )
plt.plot(snr_hist, acc_hist_iq, linewidth=9, label='Classical IQ' )
plt.xlabel("Signal to Noise Ratio", fontsize=20)
plt.ylabel("Classification Accuracy", fontsize=20)
plt.title("Classification Accuracies ", fontsize=20)
plt.legend(loc="lower right", fontsize=20)
plt.savefig('z_snr_curve_combined.png', bbox_inches='tight')