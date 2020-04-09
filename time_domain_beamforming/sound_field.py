#@version1.0 date: 01/07/2020 by zhenyuhuang
#@author:          zhenyuhuang0501@gmail.com
"""
Builder pattern to construct an acoustic enviroment
"""
from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import kron

from time_domain_beamforming.base import TimeDomainULABase
from parameters import constants  

class SoundFieldBase(ABC):
    '''
    Builder Pattern

    Basic class of adaptive beamforming acoustic enviroment:

    Suppose that a desired signal from direction phi, and two statiscally independent interferences from theta1 
    and theta2 with variance sigma1^2; And the sensor itself contains with Gaussian noise with sigma2^2
    The desired signal is a harmonic pulse of T samples 
    i_SNR is input SNR of the array
    '''
    def __init__(self, name):

        self._name = name

        #only one noise interference
        self._noise_signal_dict = {
            'interference'   : {
                'var': None,
                'direction': None
            },
            'sensor_noise_var' : None,
        }

        self._desired_signal_dict = {
            'autocorrelation'       : None,
            'phi'                   : None
        }

        self._iSNR = {
            'dB_value': None,
            'value'   : None
        }

        self._uniform_array = None


    def desired_signal_correlation(self):
        [ii, jj] = np.meshgrid(np.arange(1, self._uniform_array.L+1), np.arange(1, self._uniform_array.L+1))
        r_x = self._desired_signal_dict.get('autocorrelation') ** (np.abs(jj - ii))
        return r_x
    

    def noise_signal_correlation(self, beta=0.1):

        sigma_u_2 = self._noise_signal_dict.get('interference').get('var')
        sigma_w_2 = self._noise_signal_dict.get('sensor_noise_var')
        r_sigma_w = sigma_w_2 * np.eye(self._uniform_array.time_samples * self._uniform_array.M)

        r_sigma_u = kron(np.ones((self._uniform_array.M, self._uniform_array.M)), sigma_u_2 * np.eye(self._uniform_array.time_samples))
        r_n = r_sigma_w + r_sigma_u

        iSNR = 10 ** (self._iSNR.get('dB_value') / 10)
        n_factor = self.desired_signal_correlation()[0, 0] / r_n[0, 0] / iSNR
        r_n = n_factor * r_n
        return r_n


    def __str__(self):
        #TODO implemented for infor of sound_field  
        return 'array: Adaptive array'


class SoundFieldBuilder:
    '''
    Builder 
    '''
    def __init__(self):
        self.sound_field = SoundFieldBase('2GaussianNoise1Harmonic')


    def desired_signal(self, *, autocorrelation, phi):
        '''
        Harmonic pulse signal
        
        Parameter
        ---------
        amplitute: float
            amplitute of the harmonic signal
        '''

        self.sound_field._desired_signal_dict['autocorrelation'] = autocorrelation
        self.sound_field._desired_signal_dict['phi'] = phi

    
    def interference_signal(self, *, interference_var, interference_direction):
        '''
        Interference siganl: two IID white Gaussian noise

        Parameter
        ---------
        var: list
            noise var list of noise interferences

        directions: list
            direction of noise interferences
        '''

        self.sound_field._noise_signal_dict['interference'] = {'var':  interference_var,
                                             'direction': np.radians(interference_direction)}

    
    def sensor_noise_signal(self, *, sensor_noise_var):
        self.sound_field._noise_signal_dict['sensor_noise_var'] = sensor_noise_var


    def iSNR(self, *, iSNR_dB):
        self.sound_field._iSNR = {'dB_value': iSNR_dB,
                                  'value': 10 ** (iSNR_dB / 10)} #input array signal noise ratio


    def microphone_array(self, *, d_mic, M, time_samples=30, sinc_samples=25, fs=8000):
        phi = self.sound_field._desired_signal_dict['phi']
        self.sound_field._uniform_array = TimeDomainULABase(d_mic, M, phi, time_samples, sinc_samples, fs)




class SoundField():
    '''
    Director in build pattern
    '''
    def __init__(self):
        self.build = None

    def build_sound_field(self, *, autocorrelation, phi, interference_var,
                            interference_direction, sensor_noise_var, iSNR_dB, 
                            d_mic, M, time_samples=30, sinc_samples=25, fs=8000):

        self.build = SoundFieldBuilder()
        self.build.desired_signal(autocorrelation=autocorrelation, phi=phi)
        self.build.interference_signal(interference_var=interference_var, interference_direction=interference_direction)
        self.build.sensor_noise_signal(sensor_noise_var=sensor_noise_var)
        self.build.iSNR(iSNR_dB=iSNR_dB)
        self.build.microphone_array(d_mic=d_mic, M=M, time_samples=time_samples, sinc_samples=sinc_samples, fs=fs)

    @property
    def sound_field(self): 
        return self.build.sound_field

