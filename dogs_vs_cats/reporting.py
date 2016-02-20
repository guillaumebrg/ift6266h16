__author__ = 'Guillaume'

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from keras.models import model_from_json
from preprocessing import resize_and_scale

def draw_train_and_valid_curves(train_points, valid_points, learning_rate_updates_epoch, best_per_lr, mode="loss",
                                fignumber=0, pdf=None):
    """
    This function instantiate a figure and call the function 'plot_train_and_valid_curves'. If pdf is True, the figure
    will be printed in a pdf file.
    """

    plt.figure(fignumber, figsize=(12,8))
    plt.clf()
    ax = plt.subplot(1,1,1)
    plot_train_and_valid_curves(ax, train_points, valid_points, learning_rate_updates_epoch, best_per_lr, mode)
    if pdf is None:
        plt.show()
    else:
        pdf.savefig()
        plt.close()

def plot_train_and_valid_curves(ax, train_points, valid_points, learning_rate_updates_epoch, best_per_lr, mode="loss"):
    """
    Function used to plot loss and accuracy curves over epochs.
    """
    if mode=="loss":
        name = "Loss"
        names = "losses"
        factor = [1.2, 1.22]
        loc_legend = 1
    elif mode =="acc":
        name = "Accuracy"
        names = "acc"
        factor = [0.9, 0.88]
        loc_legend = 4
    else:
        print "Mode not understood. Available modes : 'loss' and 'acc'"
        return

    #ax = plt.subplot(1,1,1)#
    # Plot training and valid loss curves
    ax.plot(np.arange(len(train_points)),train_points, c="k", zorder=1)
    ax.plot(np.arange(len(valid_points)),valid_points, c="k", zorder=1)
    ax.scatter(np.arange(len(train_points)),train_points, c="b", label="Train %s"%names, zorder=2)
    ax.scatter(np.arange(len(valid_points)),valid_points, c="r", label="Valid %s"%names, zorder=2)
    # Plot vertical line when the learning rate was updated
    first = True
    for elem in learning_rate_updates_epoch:
        if first:
            plt.plot([elem-.5,elem-.5], [1.4*valid_points[elem],train_points[elem]*0.6], c="k", label="LR updates", linestyle="--")
            first = False
        else:
            plt.plot([elem-.5,elem-.5], [1.4*valid_points[elem],train_points[elem]*0.6], c="k", linestyle="--")
    # Plot best model in each region
    first = True
    for i,elem in enumerate(best_per_lr):
        if first:
            x = elem[0]
            y = elem[1]
            plt.scatter(x,y, c="g", label="Best models", marker="*", zorder=3, s=100)
            plt.plot([x,x],[y,factor[0]*y], c="g")
            plt.text(x,factor[1]*y, "Epoch %d"%(x), fontsize=8)
            first = False
        else:
            x = elem[0]+learning_rate_updates_epoch[i-1]
            y = elem[1]
            plt.scatter(x,y, c="g", marker="*", zorder=3, s=100)
            plt.plot()
            plt.plot([x,x],[y,factor[0]*y], c="g")
            plt.text(x,factor[1]*y, "Epoch %d"%(x), fontsize=8)
    # Xlim, Ylim, labels, legend...
    ax.set_ylim([0,1])
    ax.set_xlim([0,len(train_points)+5])
    ax.set_xlabel("Epochs")
    ax.set_ylabel(name)
    handles,labels = ax.get_legend_handles_labels()
    sorted_zip = sorted(zip([2,0,1,3],handles, labels))
    index, handles, labels = zip(*sorted_zip)
    ax.legend(handles,labels, loc=loc_legend, prop={'size':10})

def draw_perf(best_per_lr, learning_rate_updates_epoch, fignumber=0, mode="loss", pdf=None):
    """
    Call the plot_perf function. If pdf is false, a plot figure will be shown, else, the figure will be printed
    in a pdf file.
    """
    plt.figure(fignumber, figsize=(6,3))
    plt.clf()
    ax = plt.subplot(1,1,1)
    plot_perf(ax, best_per_lr, learning_rate_updates_epoch, mode)
    if pdf is None:
        plt.show()
    else:
        pdf.savefig()
        plt.close()

def plot_perf(ax, best_per_lr, learning_rate_updates_epoch, mode="loss"):
    """
    Plot performances of some models using the matplotlib function 'bar'.
    """
    colors = [ "b", "r", "g", "c", "m", "y", "k", "w"]
    ind = 2*np.arange(len(best_per_lr))
    ybars = [elem[1] for elem in best_per_lr]
    width = 1
    rect = plt.bar(ind, ybars, width, color=colors[0:len(ybars)], alpha=0.5)
    ax.set_ylim([min(ybars)*0.8,max(ybars)*1.2])
    ax.set_ylabel("Best models %s"%mode)
    ax.set_xticks(ind+width*0.5)
    tlabels = ["Epoch %d"%best_per_lr[0][0]]
    if len(best_per_lr) > 1:
        for i, elem in enumerate(best_per_lr[1:]):
            tlabels.append("Epoch %d"%(elem[0]+learning_rate_updates_epoch[i]))
    ax.set_xticklabels(tlabels)
    ax.set_yticks([])
    autolabel(ax, rect)

def autolabel(ax, rects):
    """
    Function used to add some text on a 'bar' plot.
    """
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.3f' % height,
                ha='center', va='bottom')

def draw_experiment_figures(path_to_experiment, subdir_names="MEM", history_files_name="history.pkl", pdf=None,
                            multifigures=False):
    """
    Function that take as an input the path to an experiment folder, and extract from it main results: training curves,
    performances in terms of loss and accuracy will be shown in a recap figure. This function expects a Keras training
    using the 'ModelCheckpoint_perso' class to save models.
    """
    # Get sub directories : this works only with sub dir called "MEMxxx"
    expe_list = os.listdir(path_to_experiment)
    expe_list = [path_to_experiment+"/"+dir_ for dir_ in expe_list if dir_.find(subdir_names)==0]
    # Open history.pkl files
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    for dir_ in expe_list:
        with open(dir_+"/%s"%history_files_name, "r") as f:
            train_loss.append(pickle.load(f))
            valid_loss.append(pickle.load(f))
            train_acc.append(pickle.load(f))
            valid_acc.append(pickle.load(f))
    # Concatenate those list
    # Loss
    train_loss_concat = np.concatenate(train_loss)
    valid_loss_concat = np.concatenate(valid_loss)
    # Acc
    train_acc_concat = np.concatenate(train_acc)
    valid_acc_concat = np.concatenate(valid_acc)
    # Get epochs where LR has been updated
    learning_rate_updates_epoch = np.array([len(elem) for elem in train_loss[:-1]]).cumsum()
    # Get the best model for each LR, in terms of loss
    best_per_lr_loss = [(np.argmin(elem), np.min(elem)) for elem in valid_loss]
    # Get the best model for each LR, in terms of accuracy
    best_per_lr_acc = [(np.argmax(elem), np.max(elem)) for elem in valid_acc]
    # Plot loss curves
    if multifigures: # Each figure in a different window/page
        draw_train_and_valid_curves(train_loss_concat, valid_loss_concat, learning_rate_updates_epoch,
                                    best_per_lr_loss, mode="loss", fignumber=0, pdf=pdf)
        draw_perf(best_per_lr_loss, learning_rate_updates_epoch, fignumber=1, mode="loss", pdf=pdf)
        # Plot acc curves
        draw_train_and_valid_curves(train_acc_concat, valid_acc_concat, learning_rate_updates_epoch,
                                    best_per_lr_acc, mode="acc", fignumber=2, pdf=pdf)
        draw_perf(best_per_lr_acc, learning_rate_updates_epoch, fignumber=3, mode="accuracy", pdf=pdf)
    else:
        plt.figure(figsize=(15,9))
        plt.clf()
        ax1 = plt.subplot2grid((6,6), (0,0), rowspan=3, colspan=4)
        plot_train_and_valid_curves(ax1, train_loss_concat, valid_loss_concat, learning_rate_updates_epoch,
                                    best_per_lr_loss, mode="loss")
        ax2 = plt.subplot2grid((6,6), (1,4), colspan=2)
        plot_perf(ax2, best_per_lr_loss, learning_rate_updates_epoch, mode="loss")
        ax3 = plt.subplot2grid((6,6), (3, 0), rowspan=3, colspan=4)
        plot_train_and_valid_curves(ax3, train_acc_concat, valid_acc_concat, learning_rate_updates_epoch,
                                    best_per_lr_acc, mode="acc")
        ax4 = plt.subplot2grid((6,6), (4, 4), colspan=2)
        plot_perf(ax4, best_per_lr_acc, learning_rate_updates_epoch, mode="accuracy")
        if pdf is None:
            plt.show()
        else:
            pdf.savefig()
            plt.close()

def write_experiment_report(path_to_experiment, subdir_names="MEM", history_files_name="history.pkl", multipages=False):
    """
    Function that calls the 'draw_experiment_figures' function, and store figures in a pdf. Two available options :
    one figure per page, or every figures on the same page.
    """
    if multipages is True:
        with PdfPages(path_to_experiment+'/multi_reports.pdf') as pdf:
            draw_experiment_figures(path_to_experiment, subdir_names, history_files_name, pdf=pdf, multifigures=multipages)
    else:
        with PdfPages(path_to_experiment+'/report.pdf') as pdf:
            draw_experiment_figures(path_to_experiment, subdir_names, history_files_name, pdf=pdf, multifigures=multipages)


