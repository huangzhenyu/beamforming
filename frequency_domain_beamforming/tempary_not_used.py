#@version1.0 date: 07/24/2019 by zhenyuhuang
#@version2.0 date: 11/04/2019 by zhenyuhuang
#@author:          zhenyuhuang0501@gmail.com

'''
TODO not used temporally in Frequency Domain!
This file defines the base parant class of uniform linear arrays.
Compute steer vectors/ noise coherence / plot functions
'''
import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from parameters import constants
from plot import Plot
from util import dB, hermitian, make_directory_if_not_exists


class ULABase:
    '''
    Base class of  Uniform Linear Array 

    Paramerts
    ---------
    M: int
        microphone number total 

    d_mic: float 
        distance between each microphone, unit [cm]

    phi: float 
        direction for desired signal angle , unit[degree]
    '''
    def __init__(self, d_mic, M, phi=0):

        assert(d_mic > 0)
        assert(M > 1 and isinstance(M, int))
        
        self.d_mic = d_mic
        self.M     = M
        self.phi   = np.radians(phi)

    def __str__(self):
        return 'M:{}, d:{}, phi:{} uniform microphone arrays'.format(self.M, self.d_mic, np.degrees(self.phi))

    def unit_delay(self, theta):
        '''
        computing the delay between two successive microphones

        Parameters
        ----------
        theta: float
            A given azimuth angle

        '''
        return self.d_mic / constants.get('c') * np.cos(theta)

    
    def steer_vector(self,  f, theta=None):
        '''
        Computing steer vector[导向矢量]
        
        Parameters
        ----------
        f: float 
          frequency for computing

        theta : angle list or array
        
        return: steervector when theta is list or array; single exp function when theta is a single value 
        '''
        if theta is None:
            theta = constants.get('angle_range')
            # theta  = self.phi

        if isinstance(theta, list):
            theta = np.asarray(theta)

        delay        = np.outer(self.unit_delay(theta), np.arange(self.M))
        steer_vector = np.exp(- 2j * np.pi * f * delay)

        return steer_vector.T


    def diffuse_noise_coherence(self, f, alpha=None):
        '''
        Compute the spherically isotropic(diffuse) noise field coherence matrix
        
        Parameters
        ----------
        f: float
            frequency

        alpha: float 
            diffuse factor, if the value is zero it reprensents an full isotropic duffuse sound
        '''

        [m_mat, n_mat]  = np.meshgrid(np.arange(self.M), np.arange(self.M)) 
        mat             = 2 * f * (n_mat - m_mat) * (self.d_mic / constants.get('c'))
        noise_coherence = np.sinc(mat)

        if alpha:
            noise_coherence = (1 - alpha) * noise_coherence + alpha * np.identity(self.M)

        return noise_coherence


class BeamformingBase(ULABase):

    def __init__(self, d_mic, M, phi):
        super(BeamformingBase, self).__init__(d_mic, M, phi)

        #beam type
        self.beam_type =  self.__class__.__name__

        self.beam_label = '{}_M_{}_d_{}_phi_{}'.format(self.beam_type,
                                self.M, self.d_mic, int(np.degrees(self.phi)))

    @abstractmethod
    def filter(self): 
        ''' 
        Spatial filter coefficients
        '''
        return NotImplementedError('This would be implemented'
                                        ' in children class')

        
    def beam_pattern(self, f):
        '''
        Compute and Plot beampattern response for microphone array
        
        Parameters
        ----------
        f: float
            frequency
        '''
        omega        = self.filter(f)
        steer_vector = self.steer_vector(f)
        response     = np.squeeze(dB(np.abs(hermitian(steer_vector).dot(omega))))
        return response


    def _set_fig_name(self, title):
        '''
        if save the fig, the directory and fig name should set

        Parameters
        ----------
        titile: string
            fig title name
        '''
        pic_path = os.path.join(constants.get('pic_path'), self.beam_type)
        make_directory_if_not_exists(pic_path)
        fig_name = os.path.join(pic_path ,'{}.png'.format(title))
        return fig_name


    def beam_pattern_polar(self, f, save_fig=False):

        title    = '{}_{}_f_{}'.format('Beampattern_Polar', self.beam_label, f)
        if save_fig:
            fig_name = self._set_fig_name(title)
        else:
            fig_name = None

        response = self.beam_pattern(f)
        #expand to pi to 2*pi, when plot polar

        sweep_angle = np.concatenate((constants.get('angle_range'), 
                                (constants.get('angle_range') + np.pi)), axis=0)
        response    = np.concatenate((response, np.fliplr([response])[0]), axis=0)

        plot = Plot()
        plot.append_axes(polar=True)
        plot.plot_vector(response, sweep_angle)
        plot.set_pipeline(xlim=(0, 2*np.pi), ylim=(-50, np.max(response)+1), title=title, 
                            xlabel='Azimuth Angle[Deg]', fig_name=fig_name)



    def beam_pattern_cartesian(self, f, save_fig=False):

        title    = '{}_{}_f_{}'.format('Beampattern_Cartesian', self.beam_label, f)
        if save_fig:
            fig_name = self._set_fig_name(title)
        else:
            fig_name = None

        response = self.beam_pattern(f)

        plot = Plot()
        plot.append_axes()
        plot.plot_vector(response, np.degrees(constants.get('angle_range')))
        plot.set_pipeline(xlim=(0, 180), ylim=(np.min(response)-1, np.max(response)+1), title=title,
                                xlabel='Azimuth Angle[Deg]', ylabel='Beampattern[dB]',fig_name=fig_name)
           

    def beam_pattern_cartesian_multi_freq(self, save_fig=False):
        '''
        Plot Array Response including several Frequencies and Azimuth Angle infomation
        '''
        title    = '{}_{}'.format('Beampattern_CartersianMultiFreq', self.beam_label)
        if save_fig:
            fig_name = self._set_fig_name(title)
        else:
            fig_name = None

        #frequency to plot
        freq_range  = constants.get('freq_range_small')
        angle_range = np.degrees(constants.get('angle_range'))
        y_lim_lower = []
        
        plot = Plot()
        plot.append_axes()
        for freq in freq_range:
            response_freq = self.beam_pattern(freq)
            y_lim_lower.append(np.min(response_freq))
            plot.plot_vector(response_freq, angle_range, label='{} kHz'.format(freq / 1000))
        
        y_lim_lower = np.maximum(-70, np.min(y_lim_lower))
        
        plot.set_pipeline(title=title, ylim=(y_lim_lower, 1), xlim=(0, 180), legend='lower right',
                            xlabel='Azimuth Angle[Deg]', ylabel='Beampattern[dB]', fig_name=fig_name)

        
    def beam_pattern_heatmap(self, save_fig=False):
        
        '''
        Plot heatmap of Array Response including all Frequency and Azimuth Angle infomation
        '''
        title    = '{}_{}'.format('Beampattern_Heatmap', self.beam_label)
        if save_fig:
            fig_name = self._set_fig_name(title)
        else:
            fig_name = None

        freq = np.linspace(8000, 1, num=800, endpoint=True)
        response_matrix = np.zeros((len(freq), len(constants.get('angle_range'))))

        for index, f in enumerate(freq):
            response_matrix[index:] = self.beam_pattern(f)

        #plot    
        xticks = np.arange(0, len(constants.get('angle_range')), 60)
        yticks = np.arange(0, 900, 100)
        xticklabels = np.arange(0, len(constants.get('angle_range')), 60, dtype=int) / 2.0
        yticklabels = np.linspace(8, 0, num=9, endpoint=True, dtype=int)
        
        plot = Plot()
        plot.append_axes()
        plot.plot_heatmap(response_matrix)
        plot.set_pipeline(title=title, xticks=xticks, yticks=yticks, xticklabels=[int(tick) for tick in xticklabels], 
                        yticklabels=yticklabels, xlabel='Azimuth Angle[Deg]', ylabel='Freq[kHz]', fig_name=fig_name)
