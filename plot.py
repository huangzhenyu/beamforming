#@version1.0 date: 11/23/2019 by zhenyuhuang
#@author:          zhenyuhuang0501@gmail.com
import numpy as np
import matplotlib.pyplot as plt
import _tkinter

from parameters import constants

class Plot:
    '''
    Plot class in beamforming for showing performance or comparing with others
    '''
    def __init__(self, figsize=(10, 6), subplot_rows=1, subplot_cols=1):
    
        plt.ioff()
        self.fig          = plt.figure(figsize=figsize)
        self.subplot_rows = subplot_rows
        self.subplot_cols = subplot_cols

        self.subplot_totals = subplot_rows * subplot_cols
        self.axes   = []
        self._current_axes_index = 1

    @staticmethod
    def wait_for_user_static(pause=0.2):
        if pause is None or pause == 0:
            plt.show()
        else:
            plt.pause(pause)
    

    def wait_for_user(self):
        self.wait_for_user_static(None)
    

    def append_axes(self, polar=False):
        '''
        append axes one by one to axes list when length of axes is smaller than subplot totals
        
        Parameter
        ---------
        polar: bool
            if true catesian else polar
        '''
        assert self._current_axes_index <= self.subplot_totals

        if polar:
            ax = self.fig.add_subplot(self.subplot_rows, self.subplot_cols, self._current_axes_index, polar=True)    
        else:
            ax = self.fig.add_subplot(self.subplot_rows, self.subplot_cols, self._current_axes_index)
        self.axes.append(ax)
        self._current_axes_index += 1


    def append_axes_combo(self, polar=False):
        '''
        append same axes to axes list when axes is empty continuously

        Parameter
        ---------
        polar: bool
            if true catesian else polar
        '''
        assert len(self.axes) == 0
        for fig_index in range(1, self.subplot_totals+1):
            if polar:
                self.append_axes(True)
            else:
                self.append_axes()

            

    def plot_vector(self, ydata, xdata, label=None, subplot_index=0):

        assert(subplot_index) < self._current_axes_index

        if xdata is None:
            xdata = np.arange(len(ydata))

        if label:
            self.axes[subplot_index].plot(xdata, ydata, label=label)
        else:
            self.axes[subplot_index].plot(xdata, ydata)


    def set_pipeline(self, title, xlabel=None, ylabel=None, xlim=None, ylim=None,
                    xticks=None, yticks=None, xticklabels=None, yticklabels=None,
                    legend=None, grid=True, fig_name=None, wait_for_user=True, subplot_index=0):
        '''
        pipeline setup for figure
        '''

        self.set_title(title, subplot_index = subplot_index )
        self.set_lim(xlim, ylim, subplot_index = subplot_index )
        self.set_ticks(xticks, yticks, subplot_index = subplot_index )
        self.set_ticklabels(xticklabels, yticklabels, subplot_index = subplot_index )
        self.set_label(xlabel, ylabel, subplot_index = subplot_index )
        if grid:
            self.set_grid(subplot_index = subplot_index )
        if legend is not None:
            self.set_legend(legend, subplot_index = subplot_index )
        if fig_name is not None:
            self.save_figure(fig_name)
        if wait_for_user:
            self.wait_for_user()

    def set_ticks(self, xticks, yticks, subplot_index=0):
        if xticks is not None:
            self.axes[subplot_index].set_xticks(xticks)
        if yticks is not None:
            self.axes[subplot_index].set_yticks(yticks)

    
    def set_ticklabels(self, xticklabels, yticklabels, subplot_index=0):
        if xticklabels is not None:
            self.axes[subplot_index].set_xticklabels(xticklabels)
        if yticklabels is not None:
            self.axes[subplot_index].set_yticklabels(yticklabels)
 

        
    def set_lim(self, xlim, ylim, subplot_index=0):
        if xlim:
            self.axes[subplot_index].set_xlim(xlim[0], xlim[1])
        if ylim:
            self.axes[subplot_index].set_ylim(ylim[0], ylim[1])

    
    def set_grid(self, subplot_index=0):
        self.axes[subplot_index].grid(True)

    def set_title(self, title, subplot_index=0):
        self.axes[subplot_index].set_title(title)

    def set_legend(self, loc='lower right', fontsize=6, subplot_index=0):
        self.axes[subplot_index].legend(loc=loc, fontsize=fontsize)

    def set_label(self, xlabel, ylabel, subplot_index=0):
        if xlabel:
            self.axes[subplot_index].set_xlabel(xlabel)
        if ylabel:
            self.axes[subplot_index].set_ylabel(ylabel)
    
    @staticmethod
    def save_figure_static(fig_name):
        plt.savefig(fig_name)


    def save_figure(self, fig_name):
        self.save_figure_static(fig_name)


    def plot_heatmap(self, matrix, transpose_flag=False, cmp=None, label_pad=-45, subplot_index=0):

        if transpose_flag:
            matrix = matrix.T

        if cmp is None:
            im = self.axes[subplot_index].imshow(matrix, aspect="auto")

        if subplot_index == len(self.axes) - 1:

            # clb = plt.colorbar(im)
            cb_ax = self.fig.add_axes([0.92, 0.1, 0.02, 0.75])
            clb = self.fig.colorbar(im, ax=self.axes, cax=cb_ax)
            clb.ax.set_title("[dB]")

    def suptitle(self, title, hspace=0.3):
        if self.subplot_totals >= 2:
            self.fig.suptitle(title)
            self.fig.subplots_adjust(hspace = hspace)
        else:
            raise ValueError('No such figs')