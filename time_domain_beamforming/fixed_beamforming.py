#@version1.0 date: 10/15/2019 by zhenyuhuang
#@version2.0 date: 01/07/2020 by zhenyuhuang
#@author:          zhenyuhuang0501@gmail.com
"""
Fixed Beamforming in time domain, broadband in nature.
"""
import numpy as np
from scipy.linalg import pinv, pinv2

from descriptor import LazyProperty
from parameters import constants
from time_domain_beamforming.base import TimeDomainBeamformingBase
from util import dB


class FixedBeamforming(TimeDomainBeamformingBase):
    '''
    Base class of FixedBeamforming, contain two property of TDB(Time Domain Beamforming)
    '''
    @property
    def white_noise_gain(self):
        '''
        return the white noise gain in TDB
        '''
        g = self.steer_vector()
        h = self.filter
        white_noise_gain = h.T.dot(g).dot(g.T).dot(h) / h.T.dot(h)
        white_noise_gain_db = dB(np.abs(white_noise_gain), True)
        return white_noise_gain_db  


    @property
    def directivity(self):
        '''
        return the directivity in TDB
        '''
        g = self.steer_vector()
        h = self.filter
        int_g = self.diffuse_noise_coherence()
        directivity = (h.T.dot(g.dot(g.T)).dot(h)) / (h.T.dot(int_g).dot(h))
        directivity_db = dB(np.abs(directivity), True)
        return directivity_db
   

    @property
    def front_back_ratio(self):
        '''
        return the front_back_ratio in TDB
        '''
        theta_array_0_90   = np.array_split(constants.get('angle_range'), 2)[0]
        theta_array_90_180 = np.array_split(constants.get('angle_range'), 2)[1]

        filts            = self.filter
        gamma_0_half_pi  = self.diffuse_noise_coherence(theta_array_0_90)
        gamma_half_pi_pi = self.diffuse_noise_coherence(theta_array_90_180)
        front_back_ratio = (filts.T.dot(gamma_0_half_pi).dot(filts)) / (filts.T.dot(gamma_half_pi_pi).dot(filts))
        return dB(front_back_ratio, True)


class DelayAndSum(FixedBeamforming):
    '''
    The classical delay-and-sum beamforming in the time domain is derived by maximizing
    the WNG subject to the distortionless.
    '''
    @LazyProperty
    def filter(self):
        i_ell = self.i_ell
        g_matrix = self.steer_vector()
        filts = g_matrix.dot(i_ell) / self.M
        return filts


class MaximumDF(FixedBeamforming):
    '''
    Distortionless maximum DF beamformer, minimize the denominator of the DF
    subject to the distortionless constraint in the numerator of the DF.
    '''
    @LazyProperty
    def filter(self):
        gamma = self.diffuse_noise_coherence()
        g     = self.steer_vector()
        i_ell = self.i_ell
        #pinv2 和 pinv有区别，pinv2计算奇异值多且值较大多伪逆矩阵
        filts = pinv2(gamma).dot(g).dot(pinv(g.T.dot(pinv2(gamma)).dot(g))).dot(i_ell)
        return filts


class RobustSuperDirective(FixedBeamforming):
    '''
    SuperDirective beamformer is simply a particular case of the distortionless maximum DF
    beamformer, where phi==0.
    We can derive the TDB robust superdirective beamformer.
    '''
    def __init__(self, d_mic, M, alpha=1, phi=90, time_samples=30, sinc_samples=25, fs=8000):
        super(RobustSuperDirective, self).__init__(d_mic, M, phi, time_samples, sinc_samples, fs)
        self.alpha = alpha
    

    @LazyProperty
    def filter(self):
        gamma = self.diffuse_noise_coherence() + \
                self.alpha * np.eye(self.diffuse_noise_coherence().shape[0])
        g     = self.steer_vector()
        i_ell = self.i_ell
        filts = pinv2(gamma).dot(g).dot(pinv(g.T.dot(pinv2(gamma)).dot(g))).dot(i_ell)
        return filts


class MinimumNorm(FixedBeamforming):
    '''
    Completely cancel the interference source from interference angle while 
    recovering the desired source impinging on the array from direction phi.

    MinimumNorm is obtained by maximizing the WNG while keep signal distortionless.
    '''
    def __init__(self, d_mic, M, phi=0, interference_angle=90, time_samples=30, sinc_samples=25, fs=8000):
        super(MinimumNorm, self).__init__(d_mic, M, phi, time_samples, sinc_samples, fs)
        self.C = np.c_[self.steer_vector(), self.steer_vector(np.radians(interference_angle))]
        self.I = np.r_[self.i_ell, np.zeros_like(self.i_ell)]


    @LazyProperty
    def filter(self):
       return self.C.dot(pinv(self.C.T.dot(self.C))).dot(self.I)



class NullSteering(FixedBeamforming):
    '''
    Completely cancel the interference source from interference angle while 
    recovering the desired source impinging on the array from direction phi.

    MinimumNorm is obtained by maximizing the DF while keep signal distortionless.
    '''
    def __init__(self, d_mic, M, phi=0, interference_angle=90, time_samples=15, sinc_samples=10, fs=8000):
        super(NullSteering, self).__init__(d_mic, M, phi, time_samples, sinc_samples, fs)
        self.C = np.c_[self.steer_vector(), self.steer_vector(np.radians(interference_angle))]
        self.I = np.r_[self.i_ell, np.zeros_like(self.i_ell)]


    @LazyProperty
    def filter(self):
        gamma = self.diffuse_noise_coherence()
        C = self.C
        I = self.I
        filts = pinv2(gamma).dot(C).dot(pinv2(C.T.dot(pinv2(gamma)).dot(C))).dot(I)
        return filts
