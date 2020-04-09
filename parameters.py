#@version1.0 date: 11/04/2019 by zhenyuhuang
#@author:          zhenyuhuang0501@gmail.com

'''
This file defines the main physical constants of the system
'''

import numpy as np

# tolerance for computations
eps = 1e-10

# Implement the constants as a dictionnary so that they can
# be modified at runtime.
# The class Constants gives an interface to update the value of
# constants or add new ones.
_constants = {}
_constants_default = { 
    # basic path for store kinds of picture 
    'pic_path'         : 'pic',
    # speed of sound at 20 C in dry air, unit [cm/s]
    'c'                : 34000.0,

    # large scale frequency range[1 8000] to be used for 
    # computing something changed by frequency
    'freq_range_large' : np.arange(1, 8001, 20),

    # small scale freq range to compute some specific freq beam pattern
    'freq_range_small' : np.array([500, 1000, 2000, 4000, 6000, 8000]),

    #mini scale freq range to draw plot like beam pattern polar system
    'freq_range_mini'  : np.array([1000, 2000, 4000, 8000]),

    # angle range[0, pi] to be used in plot beam 
    'angle_range'      : np.radians(np.arange(0, 181, 0.5)),

    # alpha value in first order beamformig, maybe different by different author
    'first_order_null_angle': {'Diople':         np.arccos(0) , 
                               'Cardioid':       np.arccos(-1) ,
                               'Hypercardioid':  np.arccos(-1.0 / 3.0), 
                               'Supercardioid':  np.arccos((1 - 3 ** 0.5) / (3 - 3 ** 0.5))}, 
    # alpha value in second order beamformig, maybe different by different author
    #  reference
    #  
    #  Differential Microphone Arrays P10
    #            Elmar Messner Master Thesis
    #  Those parameters are equalized for flateen frequency response.
    'second_order_null_angle': {
                               'Cardioid':       [np.arccos(-1),    np.arccos(0)] ,
                               'Hypercardioid':  [np.arccos(-0.81), np.arccos(0.31)],
                               'Supercardioid':  [np.arccos(-0.89), np.arccos(-0.28)],
                               'Quadrupole':     [np.arccos(-1 / 2 ** 0.5), np.arccos(1 / 2**0.5)]},

    # this value is wrong
    # TODO
    'third_order_null_angle': { 
                                # 'Hypercardioid': [np.arccos(-4/7), np.arccos(4/7), np.arccos(8.0/7)],
                                'Supercardioid': [np.arccos(0.217), np.arccos(0.475), np.arccos(0.286)],
                                'Cardioid':      [np.arccos(0), np.arccos(-0.5), np.arccos(-1)]
    }
    }




class Constants:
    '''
    A class to provide easy access package wide to user settable constants.
    Be careful of not using this in tight loops since it uses exceptions.
    '''

    def set(self, name, val):
        # add constant to dictionnary
        _constants[name] = val

    def get(self, name):

        try:
            v = _constants[name]
        except KeyError: 
            try:
                v = _constants_default[name]
            except KeyError:
                raise NameError(name + ': no such constant')

        return v


# the instanciation of the class
constants = Constants()

# Compute the speed of sound as a function 
# of temperature, humidity, and pressure
def calculate_speed_of_sound(t, h, p):
    '''
    Compute the speed of sound as a function of
    temperature, humidity and pressure
    Parameters
    ----------
    t: 
        temperature [Celsius]
    h: 
        relative humidity [%]
    p: 
        atmospheric pressure [kpa]
    Returns
    -------
    Speed of sound in [cm/s]
    '''

    # using crude approximation for now
    return (331.4 + 0.6*t + 0.0124*h) * 100