#@version1.0 date: 10/15/2019 by zhenyuhuang
#@version2.0 date: 01/05/2020 by zhenyuhuang
#@author:          zhenyuhuang0501@gmail.com

"""
Beamforming in the time domain, broadband in nature.
"""
import matplotlib.pyplot as plt
import numpy as np

from descriptor import LazyProperty
from parameters import constants
from plot import Plot
from uniform_linear_arrays import ULABase
from util import dB


class TimeDomainULABase(ULABase):
    """
    Base class of time domain ula

    Parameter
    ---------
    time_samples: int 
    successive time samples of the mTH sensor signal

    sinc_samples: int 
    sinc function horizontal axis. Must much larger than fs * delay(m)

    fs: sampling frequency
    """
    def __init__(self, d_mic, M, phi=90, time_samples=30, sinc_samples=25, fs=8000):
        super(TimeDomainULABase, self).__init__(d_mic, M, phi)

        self.fs = fs
        self.time_samples = time_samples # 30 in fixed, 20 in adaptive, 15 in differential
        self.sinc_samples = sinc_samples # 25 in fixed, 20 in adaptive, 10 in differential
        self.L = 2 * self.sinc_samples + self.time_samples
    

    def steer_vector(self, theta=None):
        '''
        broadband steer vector similar in frequency domain steer vector
        '''
        if theta is None:
            theta = self.phi

        [ii, jj]  = np.meshgrid(np.arange(1, self.L + 1), 
                                np.arange(1, self.time_samples + 1))
        steer_vector = np.zeros((self.M * self.time_samples, self.L))

        for n in np.arange(self.M):
            Gm_theta_d = np.sinc(1 - self.sinc_samples - self.time_samples - \
                                jj + ii - self.fs * n * self.unit_delay(theta))
            steer_vector[ (n * self.time_samples):((n+1) * self.time_samples), :] = Gm_theta_d

        return steer_vector


    def diffuse_noise_coherence(self, theta_list=constants.get('angle_range')):
        '''
        The equivalent form of diffuse noise coherence in the time domain

        Parameter
        ---------
        theta_list : array
        default, 0-pi
        '''
        g_phi = []
        for theta in theta_list:
            g = self.steer_vector(theta)
            x = g.dot(g.T)
            g_phi.append(g.dot(g.T))

        g_phi   = np.array(g_phi)
        int_g   = np.zeros((g_phi.shape[1], g_phi.shape[2]))
        d_theta = theta_list[1] - theta_list[0]
        for id1 in np.arange(g_phi.shape[1]):
            for id2 in np.arange(g_phi.shape[2]):
                int_g[id1, id2] = np.sum(g_phi[:,id1,id2] * np.sin(theta_list) * d_theta)

        return int_g / 2

    
    @LazyProperty
    def i_ell(self):
        '''
        1(Iendity matrix) in time domain to keep the output distortionless
        The position of the 1 in i_ell must coincide with the position of the maximum
        element of the diagonal of G.T.G(G is the steer vector)
        '''
        g_matrix       = self.steer_vector()
        g              = g_matrix.T.dot(g_matrix)
        g_matrix_diag  = np.diag(g)
        i_ell          = np.zeros_like(g_matrix_diag)
        idx_max        = np.argwhere(g_matrix_diag == np.max(g_matrix_diag))[0]
        i_ell[idx_max] = 1
        return i_ell
    


class TimeDomainBeamformingBase(TimeDomainULABase):

    @LazyProperty
    def filter(self): 
        """
        滤波器的系数
        """
        raise NotImplementedError('This would be implemented'
                                        ' in children class')


    def beam_pattern(self):
        """
        绘制波束形成的波束图
        """
        filts = self.filter
        performace = []
        for theta in constants.get('angle_range'):
            g_tmp = self.steer_vector(theta)
            performace_theta = filts.T.dot(g_tmp).dot(g_tmp.T).dot(filts)
            performace.append(performace_theta)

        return dB(np.array(performace), True)
        

    def beam_pattern_cartesian(self, save_fig=False):

        title    = '{}_{}'.format('Beampattern_Cartesian', self.beam_label)
        if save_fig:
            fig_name = self._set_fig_name(title)
        else:
            fig_name = None

        response = self.beam_pattern()

        plot = Plot()
        plot.append_axes()
        plot.plot_vector(response, np.degrees(constants.get('angle_range')))
        plot.set_pipeline(xlim=(0, 180), ylim=(np.min(response)-1, np.max(response)+1), title=title,
                                xlabel='Azimuth Angle[Deg]', ylabel='Beampattern[dB]',fig_name=fig_name)
