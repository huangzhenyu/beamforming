import numpy as np
from scipy.linalg import cholesky, inv, pinv, pinv2, schur

from descriptor import LazyProperty
from time_domain_beamforming.base import TimeDomainBeamformingBase
from util import dB, jeig


class AdaptiveBeamforming(TimeDomainBeamformingBase):
    '''
    Basic class of Adaptive beamforming
    
    Parameters
    ----------
    sound_filed: a SouldFieldBase class created in sound_field.py
    '''
    def __init__(self, sound_field):
        # pick value from sound_field
        phi   = sound_field._desired_signal_dict.get('phi')
        d_mic = sound_field._uniform_array.d_mic
        M     = sound_field._uniform_array.M
        time_samples = sound_field._uniform_array.time_samples
        sinc_samples = sound_field._uniform_array.sinc_samples
        fs = sound_field._uniform_array.fs
        super(AdaptiveBeamforming, self).__init__(d_mic, M, phi, time_samples, sinc_samples, fs)
        self.sound_field = sound_field


    @LazyProperty
    def performance(self):
        i       = self.i_ell
        h       = self.filter
        g       = self.steer_vector()
        r_n     = self.sound_field.noise_signal_correlation()
        r_x     = self.sound_field.desired_signal_correlation()

        oSNR    = h.T.dot(g).dot(r_x).dot(g.T).dot(h) / h.T.dot(r_n).dot(h)
        oSNR_dB = np.asscalar(dB(oSNR, True))

        nr_factor = r_n[0, 0] / (h.T.dot(r_n).dot(h))
        sigr_factor = r_x[0, 0] / (h.T.dot(g).dot(r_x).dot(g.T).dot(h))

        sigd_index = ((g.T.dot(h) - i).T.dot(r_x).dot(g.T.dot(h) - i)) / r_x[0, 0]

        performance_dict =  {'array gain'         : (oSNR_dB - self.sound_field._iSNR.get('dB_value')),
                         'noise reduction factor' : np.asscalar(dB(nr_factor, True)),
                         'signal reduction factor': dB(sigr_factor, True),
                         'signal distortion index': dB(sigd_index, True)
                        }
        return performance_dict


    @property
    def array_gain(self):
        return self.performance.get('array gain')


    @property
    def noise_reduction_factor(self):
        return self.performance.get('noise reduction factor')


    @property
    def reduction_factor(self):
        return self.performance.get('signal reduction factor')


    @property
    def signal_distortion_index(self):
        return self.performance.get('signal distortion index')


class Wiener(AdaptiveBeamforming):
    '''
    Minimum of the MSE criterion
    '''
    @property
    def filter(self):
        r_n   = self.sound_field.noise_signal_correlation()
        r_x   = self.sound_field.desired_signal_correlation()
        g     = self.steer_vector()
        filts = pinv(r_n).dot(g).dot(pinv(pinv(r_x) + g.T.dot(pinv(r_n)).dot(g))).dot(self.i_ell)
        return filts


class MVDR(AdaptiveBeamforming):
    '''
    maximum noise reduction factor while distortionless signal
    '''
    @property
    def filter(self):
        r_n   = self.sound_field.noise_signal_correlation()
        g     = self.steer_vector()
        i     = self.i_ell
        filts = pinv(r_n).dot(g).dot(pinv(g.T.dot(pinv(r_n)).dot(g))).dot(i)
        return filts
        

class Tradeoff(AdaptiveBeamforming):
    '''
    Trade off between noise reduction and desired signal distortion
    Minimize the signal distortion index with the constraint that the narrowband noise reduction factor is equal
    to a positive values that greater than 1. 
    -------------------
    Parameters: miu balance factor between signal and noise distortion
    miu = 1 which is the Wiener beamformer
    miu = 0 which is the MVDR beamformer
    miu > 1 result low residual noise at the expense of high desired signal distortion
    miu < 1 result hign residual noise and low desired signal distortion
    '''
    def __init__(self, sound_field, miu):
        phi   = sound_field._desired_signal_dict.get('phi')
        d_mic = sound_field._uniform_array.d_mic
        M     = sound_field._uniform_array.M
        time_samples = sound_field._uniform_array.time_samples
        sinc_samples = sound_field._uniform_array.sinc_samples
        fs = sound_field._uniform_array.fs
        super(AdaptiveBeamforming, self).__init__(d_mic, M, phi, time_samples, sinc_samples, fs)
        self.sound_field = sound_field
        self.miu = miu

    @property
    def filter(self):
        r_n   = self.sound_field.noise_signal_correlation()
        g     = self.steer_vector()
        r_x   = self.sound_field.desired_signal_correlation()
        i     = self.i_ell
        filts = pinv(r_n).dot(g).dot(pinv(self.miu * pinv(r_x) + g.T.dot(pinv(r_n)).dot(g))).dot(i)
        # filts = pinv(r_n).dot(g).dot(pinv(self.miu * pinv(r_x) + g.T.dot(pinv(r_n)).dot(g))).dot(i)
        return filts


class MaximumSNR(AdaptiveBeamforming):
    '''
    The eigenvector method
    the maximum eigenvalue corresponding to the maximum SNR beamformer
    '''
    @property
    def filter(self):
        r_x = self.sound_field.desired_signal_correlation()
        r_n = self.sound_field.noise_signal_correlation()
        g   = self.steer_vector()
        i   = self.i_ell

        T , v = jeig(g.dot(r_x).dot(g.T), r_n, sort=True)
        t1    = T[:, 0]
        zeta  = (t1.T.dot(g).dot(r_x).dot(i)) / (t1.T.dot((g.dot(r_x).dot(g.T)) + r_n).dot(t1))
        filts = zeta[0,0] * t1
        return filts
        

class LCMV(AdaptiveBeamforming):
    '''
    Minimize the MSE of the residual noise
    Linearly constrained minimum variance
    '''
    @property
    def filter(self):
        G_phi = self.steer_vector()
        G_phi_null = self.steer_vector(self.sound_field._noise_signal_dict.get('interference').get('direction'))
        C = np.c_[G_phi, G_phi_null]
        I = np.r_[self.i_ell, np.zeros_like(self.i_ell)]
        r_n = self.sound_field.noise_signal_correlation()
        filts = pinv2(r_n).dot(C).dot(pinv2(C.T.dot(pinv2(r_n)).dot(C))).dot(I)
        return filts
