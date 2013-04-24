#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

# creates error plot based on information
# plots mean + std deviation based on resolution
def create_plot(means, sds, res, stat, title):
    plt.figure()
    plt.axis([min(means)-max(sds)-0.1,max(means)+max(sds)+0.1,
              min(res)-1,max(res)+1])
    plt.errorbar(means, res, xerr=sds, fmt='o')
    plt.xlabel(stat)
    plt.ylabel("Resolution (Angstroms)")
    plt.title(title)
    plt.savefig(title+'.png')
    plt.close()

#create_plot([0.1,0.2,0.3,0.4],[1,1,1,1],[1,2,3,4],"yolo","test")
    
