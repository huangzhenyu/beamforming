#@version1.0 date: 11/04/2019 by zhenyuhuang
#@author:          zhenyuhuang0501@gmail.com

'''
This file contains some matrix operator or unit convert function
'''
import os
import inspect
import numpy as np
from scipy.linalg import cholesky, inv, schur
from parameters import eps


def hermitian(X):
    '''
    Compute and return Hermitian transpose
    '''
    return X.conj().T
    

def jn(n):
    '''
    Used in beampattern design
    Jacobi-Anger expansion中余弦级数展开中除去第一类贝塞尔函数的部分
    '''
    # assert isinstance(n, int)
    if n == 0:
        v = 1
    else:
        v = 2 * (1j) ** n
    return v


def jeig(A, B, sort=True):
    '''
    Joint Diagonalization Function from original Matlab script
    '''
    L = cholesky(B, lower=True) #lower triangle
    G = inv(L) 
    C = G.dot(A).dot(G.conj().T)
    [D, Q] = schur(C)
    X = G.conj().T.dot(Q)
    if sort:
        d = np.diag(D)
        D = np.diag(np.sort(d)[::-1])
        idx = np.argsort(d)[::-1]
        X = X[:,idx]
    return X, D


def dB(signal, power=False):
    '''
    Compute to dB unit for a given power
    '''
    if power is True:
        return 10 * np.log10(np.abs(signal) + eps)
    else:
        return 20 * np.log10(np.abs(signal) + eps)



def make_directory_if_not_exists(dir_name):
    '''
    Create a directory if it does not exist already.
    '''
    path = os.path.abspath(dir_name)
    if not os.path.isdir(path):
        os.makedirs(path)


def get_base_path():
    '''
    Get the root directory of beamforming.
    '''
    basepath = os.path.abspath(os.path.dirname(os.path.abspath(__file__))
                 + os.path.sep + ".")
    return basepath.replace('\\', '/')


def import_beamforming(beam_domain, beam_type, beam_name):
    '''
    import beamforming class

    Parameters
    ----------
    beam_domain: string
        beamforming domain, time or frequency
    beam_type: string
        beamforming type, such as fixed/adaptive and so on
    beam_name: string
        beamformig name, such as DelayAndSum ..
    '''
    #beam_domain and beam_type must in below strings
    assert beam_domain in ['frequency', 'time']
    assert beam_type   in ['fixed', 'adaptive', 'differential', 'pattern_design']
    
    
    beam_object = '{}_domain_beamforming.{}_beamforming'.format(beam_domain, beam_type)
    beam_file = '{}.py'.format(beam_object.replace('.', '/'))
    basepath = get_base_path()

    if os.path.isfile('{}/{}'.format(basepath, beam_file)):
        mod = __import__(beam_object, fromlist=[beam_name])
        mode_class_list = [beam[0] for beam in inspect.getmembers(mod)]
        # judge beamforming class exists
        if beam_name in mode_class_list:
            klass = getattr(mod, beam_name)
        else:
            raise ValueError('No {} beamforming class in {}!'.format(beam_name, beam_file))
    else:
        raise ValueError('No {} file!'.format(beam_file))

    return klass

if __name__ == "__main__":
    x = get_base_path()
    print(x)