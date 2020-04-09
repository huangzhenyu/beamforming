"""
Created on Oct 28 20:44:23 2019
Beamforming in the time domain, broadband in nature.
Differential Beamforming
@author: zhenyu_huang
"""

import numpy as np
from scipy.linalg import pinv2, pinv

from util import jeig
from time_domain_beamforming.fixed_beamforming import FixedBeamforming
from parameters import constants


class FirstOrderMinimumNorm(FixedBeamforming):
	'''
	MinimumNorm Beamforming. 
	
	Parameter
	---------
	phi:      desired signal angle
	phi_null: interference signal angle. 0 for dipole; 180 for cardioid
	alpha:    regularization parameter
	'''
	def __init__(self, d_mic, M, phi_null=90, phi=0, alpha=10e-5, time_samples=25, sinc_samples=10, fs=8000):
		super(FirstOrderMinimumNorm, self).__init__(d_mic, M, phi, time_samples, sinc_samples, fs)
		self.phi_null = np.radians(phi_null)
		self.alpha    = alpha


	@property
	def filter(self):
		g_phi      = self.steer_vector()
		g_phi_null = self.steer_vector(self.phi_null)
		c          = np.c_[g_phi, g_phi_null]
		i_ell      = self.i_ell
		i          = np.r_[i_ell.reshape(self.L, 1), np.zeros((self.L, 1))]
		i          = np.squeeze(i)
		filts      = c.dot(pinv(c.T.dot(c) + self.alpha * np.eye(2 * self.L))).dot(i)
		return filts


class SecondOrderMinimumNorm(FixedBeamforming):
	'''
	MinimumNorm Beamforming. 
	
	Parameter
	---------
	phi:      desired signal angle
	phi_null: interference signal angle. phi must in 90-180 scope
	alpha:    regularization parameter
	'''
	def __init__(self, d_mic, M, phi_null=[90, 180], phi=0, alpha=10e-5, time_samples=30, sinc_samples=10, fs=8000):
		super(SecondOrderMinimumNorm, self).__init__(d_mic, M, phi, time_samples, sinc_samples, fs)
		self.phi_null_1 = np.radians(phi_null[0])
		self.phi_null_2 = np.radians(phi_null[1])
		self.alpha    = alpha


	@property
	def filter(self):
		g_phi      = self.steer_vector()
		g_phi_null_1 = self.steer_vector(self.phi_null_1)
		g_phi_null_2 = self.steer_vector(self.phi_null_2)
		c          = np.c_[g_phi, g_phi_null_1, g_phi_null_2]
		i_ell      = self.i_ell
		i          = np.r_[i_ell.reshape(self.L, 1), np.zeros((self.L, 1)), np.zeros((self.L, 1))]
		i          = np.squeeze(i)
		filts      = c.dot(pinv(c.T.dot(c) + self.alpha * np.eye(3 * self.L))).dot(i)
		return filts


class ThirdOrderMinimumNorm(FixedBeamforming):
	'''
	MinimumNorm Beamforming. 
	
	Parameter
	---------
	phi:      desired signal angle
	phi_null: interference signal angle. phi must in 90-180 scope
	alpha:    regularization parameter
	'''
	def __init__(self, d_mic, M, phi_null=[90, 135, 180], phi=0, alpha=10e-5, time_samples=25, sinc_samples=10, fs=8000):
		super(ThirdOrderMinimumNorm, self).__init__(d_mic, M, phi, time_samples, sinc_samples, fs)
		self.phi_null_1 = np.radians(phi_null[0])
		self.phi_null_2 = np.radians(phi_null[1])
		self.phi_null_3 = np.radians(phi_null[2])
		self.alpha    = alpha


	@property
	def filter(self):
		g_phi      = self.steer_vector()
		g_phi_null_1 = self.steer_vector(self.phi_null_1)
		g_phi_null_2 = self.steer_vector(self.phi_null_2)
		g_phi_null_3 = self.steer_vector(self.phi_null_3)
		c          = np.c_[g_phi, g_phi_null_1, g_phi_null_2, g_phi_null_3]
		i_ell      = self.i_ell
		i          = np.r_[i_ell.reshape(self.L, 1), np.zeros((self.L, 1)), np.zeros((self.L, 1)), np.zeros((self.L, 1))]
		i          = np.squeeze(i)
		filts      = c.dot(pinv(c.T.dot(c) + self.alpha * np.eye(4 * self.L))).dot(i)
		return filts


#------------------------------------------------------------------------
# Hypercardioid keep the same as superdirective beamformer
#------------------------------------------------------------------------


class Supercadioid(FixedBeamforming):
	'''
	Derived from FBR definition
	'''
	def __init__(self, d_mic, M, phi=0, Q=1, time_samples=30, sinc_samples=25, fs=8000):
		super(Supercadioid, self).__init__(d_mic, M, phi, time_samples, sinc_samples, fs)
		self.Q = Q


	@property
	def filter(self):
		g = self.steer_vector()

		theta_array_0_90   = np.array_split(constants.get('angle_range'), 2)[0]
		theta_array_90_180 = np.array_split(constants.get('angle_range'), 2)[1]
		gamma_0_half_pi  = self.diffuse_noise_coherence(theta_array_0_90)
		gamma_half_pi_pi = self.diffuse_noise_coherence(theta_array_90_180)

		[t, _]           = jeig(gamma_0_half_pi, gamma_half_pi_pi)
		t     = t[:, 0: self.Q]

		filts = t.dot(pinv(t.T.dot(g).dot(g.T).dot(t))).dot(t.T).dot(g).dot(self.i_ell)
		return filts 


