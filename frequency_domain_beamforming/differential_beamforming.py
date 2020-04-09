#@version1.0 date: 09/18/2019 by zhenyuhuang
#@version2.0 date: 12/02/2019 by zhenyuhuang
#@author: zhenyuhuang0501@gmail.com       
'''
Differential Microphone Array.
A kind of Fixed Beamforming
Advantage:
1.Frequency invariance, which is important when dealing with broadband signals.
2.Highest gains in duffuse noise(hyper).
------------------------------------------------------------------------------

Assumption:
1.the sensor spacing is much smaller than the wavelength
2.The desired source signal propagates from endire direction(phi = 0)
'''
import numpy as np
from scipy.linalg import inv, pinv

from frequency_domain_beamforming.fixed_beamforming import FixedBeamforming
from util import hermitian, jeig
from parameters import constants


class FirstOrder(FixedBeamforming):
    '''
    First Order differential beamforming

    Parameters
    ----------
    type: string
        Diople
        Cardioid
        Hypercardioid
        Supercardioid
    '''
    def __init__(self, d_mic, pattern='Diople'):

        if pattern not in ['Diople', 'Cardioid', 'Hypercardioid', 'Supercardioid']:
            raise ValueError('No this pattern in FirstOrder Beamforming!')
        super(FirstOrder, self).__init__(d_mic, M=2, phi=0)
        self.null_directions = constants.get('first_order_null_angle')[pattern]

        self.beam_label = '{}{}'.format(pattern, self.beam_label)

    def filter(self, f):
        V = np.c_[self.steer_vector(f, self.phi), self.steer_vector(f, self.null_directions)]
        eye = np.array([1,0]).reshape(2,1)
        filter = np.dot(pinv(hermitian(V)), eye)
        return filter


class SecondOrder(FixedBeamforming):
    '''
    Second Order differential beamforming

    Parameters
    ----------
    type: string
        Cardioid(pseudo)
        Hypercardioid
        Supercardioid
        Quadrupole
    '''
    def __init__(self, d_mic, pattern='Cardioid'):
        if pattern not in ['Quadrupole', 'Cardioid', 'Hypercardioid', 'Supercardioid']:
            raise ValueError('No this pattern in SecondOrder Beamforming!')

        super(SecondOrder, self).__init__(d_mic, M=3, phi=0)
        self.null_directions = constants.get('second_order_null_angle')[pattern]
        self.beam_label = '{}{}'.format(pattern, self.beam_label)

    def filter(self, f):
        V = np.c_[self.steer_vector(f, self.phi), self.steer_vector(f, self.null_directions[0]), 
                                                        self.steer_vector(f, self.null_directions[1])]
        eye = np.array([1,0,0]).reshape(3,1)
        filter = np.dot(pinv(hermitian(V)), eye)
        return filter


class CardioidSecondOrder(FixedBeamforming):
    '''
    Perfect Second Order Cardioid differential beamforming
    Only have one null angle at 180 

    '''
    def __init__(self, d_mic):
        super(CardioidSecondOrder, self).__init__(d_mic, M=3, phi=0)
        # the only one null angle for second order differential beamforming
        self.null_directions = np.pi

    def filter(self, f):
        V = np.c_[self.steer_vector(f, self.phi), self.steer_vector(f, self.null_directions), 
                        (np.diag(np.arange(self.M)).dot(self.steer_vector(f, self.null_directions)))]
        eye = np.array([1,0,0]).reshape(3,1)
        filter = np.dot(pinv(hermitian(V)), eye)
        return filter

#--------------------------------------------------------------------------
class ThirdOrder(FixedBeamforming):
    '''
    TODO
    Second Order differential beamforming.
    The alpha value are setted in parameters.constants, but it may be wrong. 

    Reference
    ---------
    "On the Design and Implementation of Higher Order Differential Microphones"


    Parameters
    ----------
    type: string
        Cardioid      -- pseudo
        Hypercardioid -- RobustSuperDirective beamforimg in Fixedbeamforming
        Supercardioid -- Subspace method
    '''
    def __init__(self, d_mic, pattern='Hypercardioid'):
        if pattern not in [ 'Cardioid', 'Hypercardioid', 'Supercardioid']:
            raise ValueError('No this pattern in FirstOrder Beamforming!')

        super(ThirdOrder, self).__init__(d_mic, M=4, phi=0)
        self.null_directions = constants.get('third_order_null_angle')[pattern]
    
        self.beam_label = '{}{}'.format(pattern, self.beam_label)

    def filter(self, f):
        V = np.c_[self.steer_vector(f, self.phi), self.steer_vector(f, self.null_directions[0]),
                        self.steer_vector(f, self.null_directions[1]), self.steer_vector(f, self.null_directions[2])]
        eye = np.array([1,0,0,0]).reshape(4,1)
        filter = np.dot(pinv(hermitian(V)), eye)
        return filter


class CardioidThirdOrder(FixedBeamforming):
    '''
    Perfect Second Order Cardioid differential beamforming
    Only have one null angle at 180 

    '''
    def __init__(self, d_mic):
       
        super(CardioidThirdOrder, self).__init__(d_mic, M=4, phi=0)
        self.null_directions = np.pi


    def filter(self, f):
        V = np.c_[self.steer_vector(f, self.phi), self.steer_vector(f, self.null_directions), 
                    np.diag(np.arange(self.M)).dot(self.steer_vector(f, self.null_directions)), 
                        (np.diag(np.arange(self.M)) ** 2).dot(self.steer_vector(f, self.null_directions))]
        eye = np.array([1,0,0,0]).reshape(4,1)
        filter = np.dot(pinv(hermitian(V)), eye)
        return filter


class HyperCardioidThirdOrder(FixedBeamforming):
    '''
    Third Order Hyper Cardioid is SuperDirective beamformer in Fixed beamforming
    '''
    def __init__(self, d_mic):
        super(HyperCardioidThirdOrder, self).__init__(d_mic, M=4, phi=0)

    def filter(self, f):
        alpha = self.diffuse_noise_coherence(f)
        filter = self.steer_vector(f, self.phi) #phi must be 0 to fulfill cos(theta)=1
        filter = np.dot(pinv(alpha), filter) / hermitian(filter).dot(pinv(alpha)).dot(filter)
        return filter


class SuperCardioidThirdOrder(FixedBeamforming):
    '''
    TODO
    全频带的函数画不出来，在某个频率点上协方差矩阵不是正定的
    和FixedBeamforming中subspace类似，主要是cholesky分解引入
    '''
    def __init__(self, d_mic):
        super(SuperCardioidThirdOrder, self).__init__(d_mic, M=4, phi=0)

    def _gamma(self, f, alpha):
        assert isinstance(alpha, list)
        assert len(alpha) == 2
        assert alpha[1] > alpha[0]
        [m_mat, n_mat] = np.meshgrid(np.arange(self.M), np.arange(self.M))
        mat = 1j * 2 * np.pi * f * (m_mat - n_mat + np.finfo(np.float).eps) * self.unit_delay(self.phi)
        gamma = (np.exp(mat * np.cos(alpha[0])) - np.exp(mat * np.cos(alpha[1]))) / \
            mat / (np.cos(alpha[0]) - np.cos(alpha[1]))
        return gamma

    def filter(self, f):
        Gamma_0_90 = self._gamma(f, [0, np.pi/2])
        Gamma_90_180 = self._gamma(f, [np.pi/2, np.pi])
        X, _ = jeig(Gamma_0_90, Gamma_90_180)
        T1 = X[:,0]
        filter = T1 / hermitian(self.steer_vector(f, self.phi)).dot(T1)
        return filter

#-------------------------------------------------------------------
class MinimumNorm(FixedBeamforming):
    '''
    Maximation WNG to solve the unrobust differential beamforming in white noise sound field

    Parameters
    ----------
    order: int
        The minimumNorm differential beamforming order
    '''
    def __init__(self, d_mic, M, order=3, null_directions=[90, 120, 180]):

        super(MinimumNorm, self).__init__(d_mic, M, phi=0)
        self.beam_label = '{}_order_{}'.format(self.beam_label, order)

        self.null_directions_list = []
        if isinstance(null_directions, (int, float)):
            self.null_directions_list.append(np.radians(null_directions))
            self.beam_label = '{}_null_{}'.format(self.beam_label, null_directions)
        else:
            for angle in null_directions:
                self.beam_label = '{}_null_{}'.format(self.beam_label, angle)
                self.null_directions_list.append(np.radians(angle))

        self._equations_num = order + 1


    def filter(self, f):
        '''
        根据输入的null值包含的角度的个数，计算不同的阵列系数
        '''
        while len(self.null_directions_list) < (self._equations_num - 1):
            self.null_directions_list.append(self.null_directions_list[-1])
        
        s_list = [np.eye(self.M)]
        s_factor = 1
        for i in np.arange(1, len(self.null_directions_list)):
            if self.null_directions_list[i] != self.null_directions_list[i-1]:
                s_list.append(np.eye(self.M))
            else:
                s_list.append((np.diag(np.arange(self.M))) ** s_factor)
                s_factor = s_factor + 1
        
        D = np.zeros((self._equations_num, self.M), dtype=np.complex)
        D[0,:] = hermitian(self.steer_vector(f, self.phi))
    
        for i in np.arange(1, self._equations_num):
            D[i,:] = s_list[i-1].dot(self.steer_vector(f, self.null_directions_list[i-1]).conj()).T

        il     = np.zeros(self._equations_num) 
        il[0]  = 1
        il     = il.reshape(self._equations_num, 1)
        filter =  hermitian(D).dot(pinv(D.dot(hermitian(D)))).dot(il)
        return filter
