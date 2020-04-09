#@version1.0 date: 08/03/2019 by zhenyuhuang
#@version2.0 date: 11/06/2019 by zhenyuhuang
#@author:          zhenyuhuang0501@gmail.com

'''
Thie file contains severl kind of fixed beamforming.
Then defines some children class of fixed beamforming, such as delay and sum and so on
'''
import os

import numpy as np
from scipy.linalg import inv, pinv
from scipy.signal import chebwin

from plot import Plot
from frequency_domain_beamforming.base  import FrequencyDomainBeamformingBase
from parameters import constants
from util import dB, hermitian, jeig


class FixedBeamforming(FrequencyDomainBeamformingBase):
    '''
    Fixed beamforming basic fucntion to compute:
    white noise gain, directivity, and frequency reponse
    '''
    def white_noise_gain(self, f):
        '''
        For fixed beamforming, the WNG equals to  1 / w.conj().T * w
        And in delay and sum the w equals to steervector / self.N
        
        Parameters
        ----------
        f: float
            frequency
        '''
        wng = 1 / np.dot(hermitian(self.filter(f)), self.filter(f))
        return dB(wng, True)

    def directivity(self, f):
        '''
        Directivity factor of microphone array
        note: python sinc function is sinc(pi*x) / pi*x

        Parameters
        ----------
        f: float
            frequency
        '''
        
        noise_cov = self.diffuse_noise_coherence(f)
        di = 1 /  hermitian(self.filter(f)).dot(noise_cov).dot(self.filter(f))
        return dB(di, True)
    
     
    def _array_gain_freq_response(self, wng_or_df, plot=True, save_fig=False):
        '''
        plot array gain agaist freq

        Parameters
        ----------
        wng_or_df: string
            Directivity: draw directivity agaist frequency
            White_Noise_Gain: draw white_noise_gain agaist frequency
        '''
        freq_range = constants.get('freq_range_large')
        array_gain = np.zeros_like(freq_range, dtype=np.float64)

        for freq_index, freq in enumerate(freq_range):
            if   wng_or_df == 'White_Noise_Gain':
                array_gain[freq_index] = self.white_noise_gain(freq)
            elif wng_or_df == 'Directivity':
                array_gain[freq_index] = self.directivity(freq)

        if plot:
            title    = '{}_{}'.format(wng_or_df, self.beam_label)
            if save_fig:
                fig_name = self._set_fig_name(title)
            else:
                fig_name = None
            # title    = '{}_{}'.format(wng_or_df, self.title_tmp)
            # fig_name = os.path.join(self.pic_path ,'{}.png'.format(title))
            plot     = Plot()
            plot.append_axes()
            plot.plot_vector(array_gain, freq_range)
            plot.set_pipeline(title= title, xlabel='Freq[Hz]', ylabel=wng_or_df, xlim=[0, np.max(freq_range)],
                            ylim=[np.min(array_gain) - 5, np.max(array_gain) + 5],fig_name=fig_name)
            return None
        
        return array_gain

    
    def directivity_freq_response(self, plot=True, save_fig=False):
        '''
        plot directivity agaist freq 
        '''
        if plot:
            self._array_gain_freq_response('Directivity', True, save_fig)
            return None
        return self._array_gain_freq_response('Directivity', False)
    

    def white_noise_gain_freq_response(self, plot=True, save_fig=False):
        '''
        plot white_noise_gain agaist freq 
        '''
        if plot:
            self._array_gain_freq_response('White_Noise_Gain', save_fig)
            return None
        return self._array_gain_freq_response('White_Noise_Gain', False)



class DelayAndSum(FixedBeamforming):
    ''' 
    Simple Delay and Sum Beamforming
    '''

    def filter(self, f): 
        ''' 
        Beamforming filter coefficients

        Parameters
        ----------
        f: float
            frequency 
        '''
        filter = self.steer_vector(f, theta = self.phi)
        return filter / self.M


class DelayAndSumChebyWin(DelayAndSum):
    '''
    Simple Delay and Sum ULA taped window by Dolph-Chebyshev

    Parameters
    ----------

    attention_dB: float
        sidelobe amplitude attention dB
    '''

    def __init__(self, d_mic, M, phi, attenuation_dB=-50):

        super(DelayAndSumChebyWin, self).__init__(d_mic, M, phi)
        self.attenuation_dB = attenuation_dB
        self.beam_label = '{}_attenuation_{}dB'.format(self.beam_label, 
                                                np.abs(self.attenuation_dB))

    def filter(self, f):
        '''
        Beamforming filter coefficients

        Parameters
        ----------
        f:
            frequency 
        '''
        filter = super(DelayAndSumChebyWin, self).filter(f)

        #multiply a chebwin window
        filter = filter * chebwin(self.M, at=self.attenuation_dB).reshape(self.M, 1)
        return filter


class MaximumDirective(FixedBeamforming):
    '''
    Maximum Diffuse Factor Beamformer
    If set phi=0 and d_mic to very small value, 
    it is a paticular case called SuperDirective DF beamforming
    '''
    
    def filter(self, f): 
        """ 
        Compute omega of the MaxDF filter  
        ----------------------
        Parameters
        f: frequency 
        """
        filter = self.steer_vector(f, theta = self.phi)
        noise_cov_pinv = pinv(self.diffuse_noise_coherence(f))
        filter = np.dot(noise_cov_pinv, filter) / hermitian(filter).dot(noise_cov_pinv).dot(filter)
        return filter


class RobustSuperdirective(FixedBeamforming):
    '''
    Super robust directive beamforming
    

    Parameters
    ----------
    alpha: int, must 0-1
        if a = 0, it is MaximumDF
        if a = 1, it is DelayAndSum
    '''
    def __init__(self, d_mic, M, phi=0, alpha=0.1):

        if phi != 0 :
            raise ValueError("In Superdirectivy beamforming, phi must set as 0, please check it")
        if (alpha >1) or (alpha<0) :
            raise ValueError("alpha must belong 0 to 1")

        super(RobustSuperdirective, self).__init__(d_mic, M, phi)
        self.alpha = alpha
        self.beam_label = '{}_alpha_{}'.format(self.beam_label, self.alpha)

    def filter(self, f):
        filter = self.steer_vector(f, theta = self.phi)
        noise_cov_pinv = pinv(self.diffuse_noise_coherence(f, self.alpha))
        filter = np.dot(noise_cov_pinv, filter) / hermitian(filter).dot(noise_cov_pinv).dot(filter)
        return filter


class MinimumNorm(FixedBeamforming):
    '''
    Maximum White Noise Gain when muti-inteference source exits.
    
    Parameters
    ----------
    interferences : list or str (degree)
        angle of all the interferences
    '''    
    def __init__(self, d_mic, M, phi=0, interferences=[45, 90]):
        assert len(interferences) < M
        super(MinimumNorm, self).__init__(d_mic, M, phi)
        self.interferences = []
        for angle in interferences:
            self.beam_label = '{}_null_{}'.format(self.beam_label, angle)
            self.interferences.append(np.radians(angle))

    def filter(self, f):
        # constructe C Matrix
        theta = [self.phi]
        theta.extend(self.interferences) #

        C_matrix = self.steer_vector(f, theta=theta)
        i_c      = np.zeros(len(theta))
        i_c[0]   = 1

        filter = C_matrix.dot(pinv(hermitian(C_matrix).dot(C_matrix))).dot(i_c)
        return filter
            
class NullSteering(FixedBeamforming):
    '''
    Maximum DF when muti-inteference source exits. 

    Parameters
    ----------
    alpha: float 0-1
        control the balance between WNG and DF
    interferences : list or str (degree)
        angle of all the interferences
    ''' 
    def __init__(self, d_mic, M, phi=0, alpha=0.1, 
                                interferences=[45, 90]):
        super(NullSteering, self).__init__(d_mic, M, phi)
        self.alpha = alpha
        self.interferences = []

        self.beam_label = '{}_alpha_{}'.format(self.beam_label, self.alpha)
        for angle in interferences:
            self.beam_label = '{}_null_{}'.format(self.beam_label, angle)
            self.interferences.append(np.radians(angle))


    def filter(self, f):
        # constructe C Matrix
        theta = [self.phi]
        theta.extend(self.interferences) #

        C_matrix = self.steer_vector(f, theta=theta)
        i_c      = np.zeros(len(theta))
        i_c[0]   = 1

        noise_cov = self.diffuse_noise_coherence(f, self.alpha)
        filter    = pinv(noise_cov).dot(C_matrix).dot(pinv(hermitian(C_matrix).dot(pinv(noise_cov)).dot(C_matrix))).dot(i_c)
        return filter


class Subspace(FixedBeamforming):
    '''
    FIXME
    When f = 1000 or others jeig fuction is wrong. 
    '8-th leading minor of the array is not positive definite' 矩阵非正定的
    仅当subspcae等于1和M的时候和课本一致
    其他情况下计算结果与课本不一致，有待进一步研究
    
    Parameters
    ----------
    subspace: int 
        the number of subspace, 1 < subspace < M
    '''
    def __init__(self, d_mic, M, phi=0, subspace=1):

        if (subspace > M) or (subspace < 0):
            raise ValueError("Subspace must in [0, #Mic]")
        super(Subspace, self).__init__(d_mic, M, phi)
        self.subspace = subspace
        self.beam_label = '{}_Subspace_{}'.format(self.beam_label, self.subspace)


    def filter(self, f):

        filter = self.steer_vector(f, theta=self.phi)
        sig_cov = np.dot(filter.reshape(self.M, 1), filter.conj().reshape(1,self.M))
        noise_cov = self.diffuse_noise_coherence(f)
        X, D = jeig(sig_cov, noise_cov, True)

        T_matrix = X[:, 0:self.subspace]
        P_matrix = T_matrix.dot(pinv(hermitian(T_matrix).dot(T_matrix))).dot(hermitian(T_matrix))
        filter = P_matrix.dot(filter) / hermitian(filter).dot(P_matrix).dot(filter)
        return filter
