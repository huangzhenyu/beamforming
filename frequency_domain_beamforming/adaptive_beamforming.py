#@version1.0 date: 08/20/2019 by zhenyuhuang
#@version2.0 date: 12/18/2019 by zhenyuhuang
#@author:          zhenyuhuang0501@gmail.com
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import pinv
from scipy.linalg import cholesky

from descriptor import LazyProperty
from frequency_domain_beamforming.base import FrequencyDomainBeamformingBase
from parameters import constants
from util import dB, hermitian, jeig


class AdaptiveBeamforming(FrequencyDomainBeamformingBase):
    '''
    Basic class of Adaptive beamforming
    -----------------------
    Parameters:
    sound_filed: a SouldFieldBase class created in sound_field.py
    '''
    def __init__(self, sound_field):
        # pick value from sound_field
        phi   = sound_field._desired_signal_dict.get('phi')
        d_mic = sound_field._uniform_array.d_mic
        M     = sound_field._uniform_array.M
        super(AdaptiveBeamforming, self).__init__(d_mic, M, phi)

        self.sound_field = sound_field
        #temp dict storing performace value 
 

    #代理模式，LazyProperty节省资源开销，当只有需要当时候才生成此特性。
    @LazyProperty
    def broadband_performance_dict(self):
        '''
        Compute the preformace value for an adaptive array.

        Return
        ------
        A dict contains 'array gain', 
                        'noise reduction factor', 
                        'signal reduction facor', 
                        'signal distortion index' values
        '''
        #数字信号宽带计算频率，计算自适应Beamformer的一些性质（宽带的）
        freq = np.linspace(-0.5, 0.5, num=1000, endpoint=True) * 1 / self.sound_field.Ts
        freq = freq[1:-1]

        # 初始化一些矩阵
        filter_vec          = np.zeros((self.M, len(freq)), dtype=np.complex)  #滤波器系数
        steer_vec           = np.zeros((self.M, len(freq)), dtype=np.complex)  #导向矢量
        
        observed_cov_vec    = np.zeros((self.M, self.M, len(freq)), dtype=np.complex) #观测矩阵相关系数
        noise_cov_vec       = np.zeros((self.M, self.M, len(freq)), dtype=np.complex) #噪声的相关系数矩阵

        desired_var_vec     = np.zeros(len(freq), dtype=np.complex)                       #期望信号方差
        noise_var_vec       = self.sound_field.noise_signal_var * np.ones(len(freq))  #第一个阵元噪声的方差
 
        #赋值
        for i, f in enumerate(freq):
            filter_vec[:,i]         = self.filter(f).T
            steer_vec[:,i]          = self.steer_vector(f, self.phi).T
            desired_var_vec[i]      = self.sound_field.desired_signal_var(f)
            observed_cov_vec[:,:,i] = self.sound_field.observed_signal_cov(f)
            noise_cov_vec[:,:,i]    = self.sound_field.noise_signal_cov(f)

        # 输出的信号和噪声的能量
        output_signal_energy =  desired_var_vec.dot(np.abs(np.sum(filter_vec.conj() * steer_vec, axis=0)) ** 2) #宽带输出信号能量
        output_noise_energy  = 0
        #equations-8
        for i,f in enumerate(freq):
            output_noise_energy += hermitian(filter_vec[:,i]).dot(noise_cov_vec[:,:,i]).dot(filter_vec[:,i])    #宽带输出噪声能量

        # 属性字典 
        performance_dict = {'array gain'             : dB(output_signal_energy / output_noise_energy, power=True) \
                                                                - self.sound_field._iSNR.get('dB_value'),
                                  'noise reduction factor' : dB(np.sum(noise_var_vec) / output_noise_energy, power=True),
                                  'signal reduction factor': dB(np.sum(desired_var_vec) / output_signal_energy, power=True),
                                  'signal distortion index': dB((desired_var_vec.dot(np.abs(np.sum(filter_vec.conj() * \
                                                        steer_vec, axis=0) - 1) ** 2)) / np.sum(desired_var_vec), power=True)
                        }
        return performance_dict


    @property
    def array_gain(self):
        return self.broadband_performance_dict.get('array gain')


    @property
    def noise_reduction_factor(self):
        return self.broadband_performance_dict.get('noise reduction factor')


    @property
    def signal_reduction_factor(self):
        return self.broadband_performance_dict.get('signal reduction factor')


    @property
    def signal_distortion_index(self):
        return self.broadband_performance_dict.get('signal distortion index')


class Wiener(AdaptiveBeamforming):
    '''
    minimize the narrowband MSE(mean square error), the error signal between
    the estimated and desired-signals at a certain frequency.This part contains 
    desired signal distortion and residual noise.

    At last, the Wiener filter can be expressed as a fuction of the statics of the 
    observation and noise signals covariance matrix.
    '''
    def filter(self, f):
        observed_cov = self.sound_field.observed_signal_cov(f)
        noise_cov    = self.sound_field.noise_signal_cov(f)
        I_M          = np.eye(self.M)
        filter       = (I_M - pinv(observed_cov).dot(noise_cov)).dot(I_M[:,0])
        return filter


class MVDR(AdaptiveBeamforming):
    '''
    Minimum-Variance-Distortionless-Response
    minimize the narrowband MSE of the signal residual noise subject to distortionless constraint.
    It depends on the statics of the observations only, which can be easily estimated in practice.

    In simulation the result of noise variance is 0, 
    but because of reverberation it is not true in real.
    '''
    def filter(self, f):
        d            = self.steer_vector(f, self.phi)
        observed_cov = self.sound_field.observed_signal_cov(f)
        filter       = pinv(observed_cov).dot(d) / hermitian(d).dot(pinv(observed_cov)).dot(d)
        return filter


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
    def __init__(self, sound_field, *, miu=0.5):
        super(Tradeoff, self).__init__(sound_field)
        self.miu = miu
        self.beam_label = '{}_miu_{}'.format(self.beam_label, self.miu)

    def filter(self, f):
        observed_cov = self.sound_field.observed_signal_cov(f)
        noise_cov    = self.sound_field.noise_signal_cov(f)
        desired_var  = self.sound_field.desired_signal_var(f)
        filter       = desired_var * (pinv(observed_cov + self.miu * noise_cov)).dot(self.steer_vector(f, self.phi))
        return filter


class LCMV(AdaptiveBeamforming):
    '''
    Linearly constrained minimum variance
    Minimize the narrowband MSE of residual noise subject to theta_d distortionless
    and theta_interference null.
    The same beamformer in Fixedbeamforming.Nullsteering
    '''
    def filter(self, f):
        observed_cov = self.sound_field.observed_signal_cov(f)
        theta  = [self.phi, *self.sound_field._noise_signal_dict.get('interferences').get('directions')]
        C      = self.steer_vector(f, theta=theta)
        i_c    = np.zeros(len(theta))
        i_c[0] = 1
        filter  = pinv(observed_cov).dot(C.dot(pinv(hermitian(C).dot(pinv(observed_cov)).dot(C))).dot(i_c))
        return filter


class Estimator:
    '''
    Class of Estimation which may used in Adaptiver filter：
    1. SNR method complex
    2. SNR direct method
    3. DOA music

    -----------------------
    Parameters:
    sound_filed: a SouldFieldBase class created in sound_field.py
    alpha : noise enviroment banlance factor between white noise and diffuse noise envoroment
    ksanp : snpashot in estimator a stastics property of an acoustic soundfield
    '''
    def __init__(self, sound_field, alpha, ksnap=None):
        self.sound_field = sound_field
        self.alpha = alpha
        self.M = self.sound_field._uniform_array.M
        self.phi = self.sound_field._uniform_array.phi
        self.ksnap = ksnap


    def steer_vector(self, f, theta):
        return self.sound_field._uniform_array.steer_vector(f, theta)


    # 归一化之后的
    def noise_coherence(self, f):
        return self.sound_field._uniform_array.diffuse_noise_coherence(f, self.alpha)


    # 归一化之后的
    def isotropic_noise_coherence(self, f):
        return self.sound_field._uniform_array.diffuse_noise_coherence(f)


    def desired_signal_var(self, f):
        return self.sound_field.desired_signal_var(f)


    def noise_signal_cov(self, f):
        '''
        计算噪声的协方差矩阵，数值仿真中需要计算，但实际使用的时候得设法估计噪声的统计特性
        '''
        noise_coherence = self.noise_coherence(f)
        desired_var     = self.desired_signal_var(f)
        sigma2          = desired_var / self.sound_field._iSNR.get('value') / noise_coherence[0,0]

        if self.ksnap:
            C = cholesky(noise_coherence)
            np.random.seed(1253)
            V0 = np.dot(hermitian(C), np.random.randn(int(self.M), int(self.ksnap)))
            V = sigma2 ** 0.5 * V0
            noise_cov = np.dot(V, hermitian(V)) / self.ksnap
        else:
            noise_cov = sigma2 * noise_coherence

        return noise_cov


    def desired_signal_cov(self, f):
        d = self.steer_vector(f, self.phi)
        observed_cov = self.noise_signal_cov(f) + self.desired_signal_var(f) *\
                         np.dot(d.reshape(self.M, 1), d.conj().reshape(1, self.M))
        return observed_cov


    def observed_signal_gamma(self, f):
        '''
        TODO
        以上都用不到，实际情况只要估计这个统计量
        '''
        return self.desired_signal_cov(f) / self.desired_signal_cov(f)[0, 0]
    

    def snr_complex_method(self, f):
        '''
        averege method by real and imag decompostion
        '''
        TmpH1 = (np.real(self.observed_signal_gamma(f)) - self.isotropic_noise_coherence(f)) / \
                (np.real(np.dot(self.steer_vector(f, self.phi).reshape(self.M, 1), 
                self.steer_vector(f, self.phi).conj().reshape(1,self.M))) - self.isotropic_noise_coherence(f))

        TmpH2 = np.imag(self.observed_signal_gamma(f)) / np.imag(np.dot(self.steer_vector(f, self.phi).reshape(self.M, 1), 
                self.steer_vector(f, self.phi).conj().reshape(1, self.M)))
        
        [m_mat, n_mat] = np.meshgrid(np.arange(self.M), np.arange(self.M))
        idx = np.where((m_mat - n_mat)>0)
        return  dB(np.complex(np.sum(TmpH1[idx] + TmpH2[idx]) / self.M / (self.M-1)))


    def snr_direct_method(self, f):
        '''
        Estimate the single-channel Wiener gain directly
        '''
        hw = hermitian(self.steer_vector(f, self.phi)).dot(self.observed_signal_gamma(f) - \
                self.isotropic_noise_coherence(f)).dot(self.steer_vector(f, self.phi)) / \
                (self.M ** 2 - hermitian(self.steer_vector(f, self.phi)).dot(self.isotropic_noise_coherence(f)).dot(self.steer_vector(f, self.phi)))
        return dB(hw)[0][0]


    def doa_music(self, f):
        '''
        利用导向矢量和子空间正交基正交性质来计算空间谱, 可扩展到宽带情况
        '''
        observed_gamma = self.observed_signal_gamma(f)
        noise_gamma    = self.isotropic_noise_coherence(f) / self.isotropic_noise_coherence(f)[0, 0]

        #联合对角化
        T, _ = jeig(observed_gamma, noise_gamma)

        theta = np.arange(40, 105, 5)#计算空间谱的角度分辨率
        space_spectrum = np.zeros_like(theta, dtype=np.float32)
        for i, t in enumerate(theta):
            d_theta = self.steer_vector(f, np.radians(t))
            for m in np.arange(1, self.M):
                space_spectrum[i] = space_spectrum[i] + \
                     np.abs(T[:, m].conj().T.dot(d_theta)) ** 2

        plt.figure(figsize=(8,6))
        plt.plot(theta, space_spectrum, marker ="o")
        plt.plot(theta, space_spectrum, color = "black")
        plt.grid()
        plt.xlim([40,100])
        plt.ylim([0, 12])

        plt.title("DOA_M_{},Estimator phi_{}".format(self.M, theta[np.argmin(space_spectrum)]))
        plt.xlabel("Azimuth Angle[Degree]")
        plt.ylabel("Angular Spectrum")
        plt.show()

        return theta[np.argmin(space_spectrum)]


