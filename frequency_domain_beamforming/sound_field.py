#@version1.0 date: 12/16/2019 by zhenyuhuang
#@author:          zhenyuhuang0501@gmail.com
"""
Builder pattern to construct an acoustic enviroment
"""
from abc import ABC, abstractmethod

import numpy as np

from frequency_domain_beamforming.base import FrequencyDomainULABase
from parameters import constants

# class ULA()

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

        self._noise_signal_dict = {
            'interferences'   : {
                'vars': [],
                'directions':[]
            },
            'sesor_noise_var' : None,
        }

        self._desired_signal_dict = {
            'f0'              : None,
            'samples'         : None,
            'amplitude'       : None,
            'phi'             : None
        }

        self._iSNR = {
            'dB_value': None,
            'value'   : None
        }

        self._uniform_array = None

        # sampling internal
        
        self.Ts =  None


    def __noise_signal_cov_before_amp(self, f):
        '''
        The correlation matrix of noise before amplituded which iSNR sigma introduce.  PHI_V0
        '''
        #list of directions interferences
        var_list = self._noise_signal_dict.get('interferences').get('vars')
        steer_vector_list = []
        for phi_val in self._noise_signal_dict.get('interferences').get('directions'):
            steer_vector_list.append(self._uniform_array.steer_vector(f, phi_val))

        noise_cov = self._desired_signal_dict.get('samples') * self._noise_signal_dict.get('sensor_noise_var') *\
                         np.eye(self._uniform_array.M, dtype=np.complex)

        for s_vec, var_value in zip(steer_vector_list, var_list):
            sigma = self._desired_signal_dict.get('samples') * var_value * np.dot(s_vec.reshape(self._uniform_array.M, 1), 
                                                            s_vec.conj().reshape(1, self._uniform_array.M))
            noise_cov += sigma
        return noise_cov
    

    def noise_signal_cov(self, f):
        '''
        The correlation matrix of noise.
        '''
        return self.noise_signal_var / self.__noise_signal_cov_before_amp(f)[0, 0] * self.__noise_signal_cov_before_amp(f)


    def desired_signal_var(self, f):
        '''
        The energy of the desired signal. phi_x1
        '''
        f = f  * self.Ts
        f0 = self._desired_signal_dict.get('f0') * self.Ts
        #设计信号的方差
        desired_var = self._desired_signal_dict.get('amplitude') ** 2 / 4*(np.sin(self._desired_signal_dict.get('samples') *\
                             np.pi * (f + f0)) / np.sin(np.pi * (f + f0))) ** 2 + self._desired_signal_dict.get('amplitude') ** \
                            2 / 4*(np.sin(self._desired_signal_dict.get('samples') * np.pi * (f - f0 + np.finfo(float).eps)) / \
                            np.sin(np.pi * (f - f0 + np.finfo(float).eps))) ** 2
        return desired_var


    @property
    def noise_signal_var(self):
        '''
        噪声的增益，也即方差, phi_v1
        '''
        return self._desired_signal_dict.get('samples') * self._desired_signal_dict.get('amplitude') ** \
                2 / 2 / self._iSNR.get('value')


    def observed_signal_cov(self, f):
        '''
        观测信号的协方差矩阵，phi_Y
        '''
        d = self._uniform_array.steer_vector(f, self._uniform_array.phi)
        observed_cov = self.noise_signal_cov(f) + self.desired_signal_var(f) \
                        * np.dot(d.reshape(self._uniform_array.M, 1), d.conj().reshape(1, self._uniform_array.M))
        return observed_cov


    def __str__(self):
        #TODO implemented for infor of sound_field  
        return 'array: {}\n'.format(self._uniform_array) + \
                'desired_signal: f0 {}, A {}\n'.format(self._desired_signal_dict.get('f0'), 
                                                self._desired_signal_dict.get('amplitude'))


class SoundFieldBuilder:
    '''
    Builder 
    '''
    def __init__(self):
        self.sound_field = SoundFieldBase('2GaussianNoise1Harmonic')


    def desired_signal(self, *, f0, samples, phi, amplitude):
        '''
        Harmonic pulse signal
        
        Parameter
        ---------
        f0: float or int 
            frequency of the harmonic
        
        samples: int
            samples number total

        amplitute: float
            amplitute of the harmonic signal
        '''
        self.sound_field._desired_signal_dict['f0'] = f0
        self.sound_field._desired_signal_dict['samples'] = samples
        self.sound_field._desired_signal_dict['amplitude'] = amplitude
        self.sound_field._desired_signal_dict['phi'] = phi

    
    def interference_signal(self, *, interferences_var, interferences_direction):
        '''
        Interference siganl: two IID white Gaussian noise

        Parameter
        ---------
        var: list
            noise var list of noise interferences

        directions: list
            direction of noise interferences
        '''

        self.sound_field._noise_signal_dict['interferences'] = {'vars':        interferences_var,
                                             'directions': [np.radians(theta) for theta in interferences_direction]}

    
    def sensor_noise_signal(self, *, sensor_noise_var):
        self.sound_field._noise_signal_dict['sensor_noise_var'] = sensor_noise_var


    def iSNR(self, *, iSNR_dB):
        self.sound_field._iSNR = {'dB_value': iSNR_dB,
                                  'value': 10 ** (iSNR_dB / 10)} #input array signal noise ratio


    def microphone_array(self, *, d_mic, M):
        phi = self.sound_field._desired_signal_dict['phi']
        self.sound_field._uniform_array = FrequencyDomainULABase(d_mic, M, phi)
        self.sound_field.Ts = self.sound_field._uniform_array.d_mic / constants.get('c')



class SoundField():
    '''
    Director in build pattern
    '''
    def __init__(self):
        self.build = None

    def build_sound_field(self, *, f0, samples, phi, amplitude, interferences_var,
                            interferences_direction, sensor_noise_var, iSNR_dB, d_mic, M):

        self.build = SoundFieldBuilder()
        self.build.desired_signal(f0=f0, samples=samples, phi=phi, amplitude=amplitude)
        self.build.interference_signal(interferences_var=interferences_var, interferences_direction=interferences_direction)
        self.build.sensor_noise_signal(sensor_noise_var=sensor_noise_var)
        self.build.iSNR(iSNR_dB=iSNR_dB)
        self.build.microphone_array(d_mic=d_mic, M=M)

    @property
    def sound_field(self): 
        return self.build.sound_field


if __name__ == "__main__":

    sd = SoundFieldDirector()
    sd.build_sound_field(1000, 512, 0, 0.5, [1,1], [50, 30], 1, 0)
