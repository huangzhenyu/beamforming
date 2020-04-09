#@version1.0 date: 10/09/2019 by zhenyuhuang
#@version2.0 data: 12/07/2019 by zhenyuhuang
#@author:          zhenyuhuang0501@gmail.com
"""
Beampattern Design
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import inv, pinv, toeplitz
from scipy.special import iv, jv

from frequency_domain_beamforming.fixed_beamforming import FixedBeamforming
from util import hermitian, jn


class BeamPatternDesign(FixedBeamforming):
    '''
    Base class of all the beampattern design beamforming.

    设计的beampattern目标形式的函数是用角度不同倍数的余弦函数的加权和，alpha正是这些加权系数

    Parameter
    ---------
    order:  the order of beampattern what we designed
    alpha: list
           设计的beampattern目标形式的函数是用角度不同倍数的余弦函数的加权和逼近的，alpha正是这些加权系数
    '''
    def __init__(self, d_mic, M, order, alpha):
        assert(len(alpha) == order + 1)
        super(BeamPatternDesign, self).__init__(d_mic, M, phi=0)
        self.order = order
        self.beam_label = '{}_order_{}'.format(self.beam_label, self.order)
        for val in alpha:
            self.beam_label = '{}_alpha_{}'.format(self.beam_label, val)

        self.alpha = np.array(alpha).reshape(self.order + 1, 1)   
    

    def _fm(self, f):
        '''
        e指数中除去余弦值和虚数的部分
        '''
        m = np.arange(self.M)
        return 2 * np.pi * f * self.unit_delay(self.phi) * m


    def b_matrix_left(self, f):
        '''
        计算阵列系数的左乘矩阵, 等号右边的常数矩阵是alpha。 

        equations(23/26)
        '''
        mat = np.zeros((self.order + 1, self.M), dtype=np.complex)
        for i in np.arange(self.order + 1):
            mat[i, :] = jn(i) * jv(i, self._fm(f))
        
        return mat


class GammaMixin():
    '''
    插件类, 扩展类的功能
    '''
    def _gamma(self, f, epsilon):
        '''
        frequecy-invariant beampattern 目标函数的计算引入的矩阵

        Parameters

        f : float or int
            frequency
        
        episilon: int or float must be >= 0
            regularization prameter to do a better compromise with white noise amplification
        '''
        [m_mat, n_mat] = np.meshgrid(np.arange(self.M), np.arange(self.M))
        mat = 2j * np.pi * f * (n_mat - m_mat) * self.unit_delay(self.phi)
        gamma =   iv(0, mat) + epsilon * np.eye(self.M)
        return gamma
    

    def _gamma_dpc(self, f):
        '''
        Least squares method 计算error时候引入的矩阵，计算方法可参考
        Array processing Page364 equation10-16

        Parameter
        ---------
        f : float or int 
        '''
        fm = self._fm(f)
        gamma_dpc = np.zeros((self.order + 1, self.M), dtype=np.complex)
        for i in np.arange(self.order + 1):
            gamma_dpc[i, :] = jv(i, -fm) * (1j) ** i
        return gamma_dpc.T



class NonRobust(BeamPatternDesign):
    '''
    Nonrubust 

    Design Example
    --------------
    FirstOrder M = 2
    if 'Diople',       alpha [0 ,1]
    if 'Cardioid',     alpha [0.5, 0.5]
    if 'HyperCardioid' alpha [0.25, 0.75]
    if 'SuperCardioid' alpha [0.366, 0.634]
    '''
    def __init__(self, d_mic, M, order, alpha):
        super(NonRobust, self).__init__(d_mic, M, order, alpha)

        #nonrubust must fullfill this condition
        assert(M == order + 1)  
        

    def filter(self, f):
        B = self.b_matrix_left(f)
        filter = inv(B).dot(self.alpha)
        return filter

#-----------------------------Frequency Invariant--------------------------------------#

class Robust(BeamPatternDesign):
    '''
    Mininorm解，优化白噪声增益，使得阵列更Robust。
    不同于Nonrobust，麦克风个数必须等于阶数加一（alpha长度），此阵列麦克风个数必须大于阶数加一。

    Design Example
    --------------
    FirstOrder M can be either number larger than 2
    if 'Diople',       alpha [0 ,1]
    if 'Cardioid',     alpha [0.5, 0.5]
    if 'HyperCardioid' alpha [0.25, 0.75]
    if 'SuperCardioid' alpha [0.366, 0.634]
    '''

    def filter(self, f):
        B = self.b_matrix_left(f)
        filter = hermitian(B).dot(inv(B.dot(hermitian(B)))).dot(self.alpha)
        return filter


class FreqInvariant(BeamPatternDesign, GammaMixin):
    '''
    频率无关的一阶差分波束形成
    低频不太稳定，高频计算结果一致，但是低于1000，差异较大(WNG or DF)。
    Fixed: 求逆由inv改用pinv伪逆，效果会好很多。

    Design Example
    --------------
    FirstOrder M can be either number larger than or equals to 2
    if 'Diople',       alpha [0 ,1]
    if 'Cardioid',     alpha [0.5, 0.5]
    if 'HyperCardioid' alpha [0.25, 0.75]
    if 'SuperCardioid' alpha [0.366, 0.634]

    Parameter
    ---------
    episilon: int or float must be >= 0
        regularization prameter to do a better compromise with white noise amplification
    '''
    def __init__(self, d_mic, M, order, epsilon, alpha):
        super(FreqInvariant, self).__init__(d_mic, M, order, alpha)

        self.epsilon = epsilon
        self.beam_label = '{}_epsilon_{}'.format(self.beam_label, self.epsilon)

    
    def filter(self, f):
        B = self.b_matrix_left(f)
        C = pinv(self._gamma(f, self.epsilon))
        A = hermitian(B)
        filter = C.dot(A).dot(pinv(B.dot(C).dot(A))).dot(self.alpha)
        return filter

#-----------------------------Least Squares Method------------------------------------------#

class RegularizedLeastSquares(BeamPatternDesign, GammaMixin):
    '''
    最小均方差法则，使得目标形状和设计形状误差最小

    Design Example
    --------------
    FirstOrder M can be either number larger than or equals to 2
    if 'Diople',       alpha [0 ,1]
    if 'Cardioid',     alpha [0.5, 0.5]
    if 'HyperCardioid' alpha [0.25, 0.75]
    if 'SuperCardioid' alpha [0.366, 0.634]

    Parameter
    ---------
    episilon: int or float must be >= 0
        regularization prameter to do a better compromise with white noise amplification
    '''
    def __init__(self, d_mic, M, order, epsilon, alpha):
        super(RegularizedLeastSquares, self).__init__(d_mic, M, order, alpha)
        self.epsilon = epsilon
        self.beam_label = '{}_epsilon_{}'.format(self.beam_label, self.epsilon)

    def filter(self, f):
        C = pinv(self._gamma(f, self.epsilon))
        B = self._gamma_dpc(f)
        filter = C.dot(B).dot(self.alpha)
        return filter

class ConstrainedLeastSquares(RegularizedLeastSquares):
    '''
    RegularLeastSquares类的一个子类，重载filter方法
    在endfire方向不失真的情况下达成最小均方误差最小的设计

    Design Example
    --------------
    FirstOrder M can be either number larger than or equals to 2
    if 'Diople',       alpha [0 ,1]
    if 'Cardioid',     alpha [0.5, 0.5]
    if 'HyperCardioid' alpha [0.25, 0.75]
    if 'SuperCardioid' alpha [0.366, 0.634]

    Parameter
    ---------
    episilon: int or float must be >= 0
        regularization prameter to do a better compromise with white noise amplification
    '''

    def filter(self, f):
        filts_ls = super(RegularizedConstrainedLeastSquares, self).filter(f)
        d = self.steer_vector(f, self.phi)
        d_h = hermitian(d)
        C = pinv(self._gamma(f, self.epsilon))
        filter = filts_ls - ((1 - d_h.dot(filts_ls)) / d_h.dot(C).dot(d)) * C.dot(d)
        return filter
    
#----------------------------Joint Optimization------------------------------------------#

class ToeplitzMixin():
    '''
    联合方法中拖普里兹矩阵的构建
    '''
    def _h_matrix(self, f):
        filter_nr = NonRobust(self.d_mic, self.order+1, self.order, self.alpha).filter(f)
        A = np.concatenate((np.conj([filter_nr[0]]), np.zeros((self.M -2, 1))), axis=0)
        B = np.concatenate((hermitian(filter_nr), np.zeros((1, self.M -2))), axis=1)
        return hermitian(toeplitz(A,B))


class JointMaxWhiteNoiseGain(BeamPatternDesign, ToeplitzMixin):
    '''
    通过数学技巧，构造了一个矩阵，使得波束的设计多加一个限定条件，在endfire方向不失真同时最大化WNG
    '''

    def filter(self, f):
        H = self._h_matrix(f)
        dt = hermitian(H).dot(self.steer_vector(f, self.phi))
        g_mwng = pinv(hermitian(H).dot(H)).dot(dt) /  hermitian(dt).dot(pinv(hermitian(H).dot(H))).dot(dt)
        filter = H.dot(g_mwng)
        return filter

class JointConstrainedLeastSquares(BeamPatternDesign, ToeplitzMixin, GammaMixin):
    '''
    在endfire方向不失真同时最小均方误差最小
    '''
    def filter(self, f):
        H = self._h_matrix(f)
        dt = hermitian(H).dot(self.steer_vector(f, self.phi))
        R =  hermitian(H).dot(self._gamma(f, epsilon = 0)).dot(H)
        g_Ls = pinv(R).dot(hermitian(H)).dot(self._gamma_dpc(f)).dot(self.alpha)
        g_Cls = g_Ls + ((1 -  hermitian(dt).dot(g_Ls)) \
            / (hermitian(dt).dot(pinv(R)).dot(dt))) * (pinv(R)).dot(dt)
        filter = H.dot(g_Cls)
        return filter


class JointTradeoff(BeamPatternDesign, ToeplitzMixin, GammaMixin):
    '''
    MAXWNG和最小化ERROR的TRADEOFF

    Parameter
    ---------
    mu: float [0, 1]
        trade off between wng and error beampattern more robust agaist white noise amplification, 
        but leads to slightly less freq-invariant response
    '''
    def __init__(self, d_mic,  M, order, alpha, mu):
        super(JointTradeoff, self).__init__(d_mic, M, order, alpha)
        self.mu = mu
        self.beam_label = '{}_mu_{}'.format(self.beam_label, self.mu)
        
    
    def filter(self, f):
        H = self._h_matrix(f)
        dt = hermitian(H).dot(self.steer_vector(f, self.phi))
        R =  hermitian(H).dot(self._gamma(f, epsilon = 0)).dot(H)

        R_mu = self.mu * R + (1 - self.mu) * hermitian(H).dot(H)
        G_u = self.mu * pinv(R_mu).dot(hermitian(H)).dot(self._gamma_dpc(f)).dot(self.alpha)
        G_t = G_u + ((1 - hermitian(dt).dot(G_u)) / (hermitian(dt).dot(inv(R_mu).dot(dt)))) * pinv(R_mu).dot(dt)
        filter = H.dot(G_t)
        return filter



# if __name__ == "__main__":

#     MN = CLSFirsrOrderSuperTradeoff(0.5, 6, 0.99)
#     # f = MN.beamPattern(1000, "Polar")
#     # print(MN.filts(1000))
#     # MN.directivityFreqResponse()
#     MN.whiteNoiseGainFreqResponse()
