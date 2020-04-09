#@version1.0 date: 07/24/2019 by zhenyuhuang
#@version2.0 date: 11/04/2019 by zhenyuhuang
#@version3.0 date: 01/04/2020 by zhenyuhuang
#@author:          zhenyuhuang0501@gmail.com

'''
This file defines the base parant class of uniform linear arrays.
Some basic information of a microphone array: 
        Mic numbers/distance/desired signal wav incident angle
        Unit delay
        Abstract method of steer vector and diffuse noise coherence
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

        self.beam_type =  self.__class__.__name__

        self.beam_label = '{}_M_{}_d_{}_phi_{}'.format(self.beam_type,
                                self.M, self.d_mic, int(np.degrees(self.phi)))

    def __str__(self):
        return self.beam_label

    
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


    def unit_delay(self, theta):
        '''
        computing the delay between two successive microphones

        Parameters
        ----------
        theta: float
            A given azimuth angle

        '''
        return self.d_mic / constants.get('c') * np.cos(theta)

    
    @abstractmethod
    def steer_vector(self):
        '''
        Computing steer vector[导向矢量]
        '''

        return NotImplementedError('This would be implemented'
                                        ' in children class')


    @abstractmethod
    def diffuse_noise_coherence(self, f, alpha=None):
        '''
        Compute the spherically isotropic(diffuse) noise field coherence matrix
        '''

        return NotImplementedError('This would be implemented'
                                        ' in children class')

