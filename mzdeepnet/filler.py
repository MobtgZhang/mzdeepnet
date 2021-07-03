import numpy as np
class Filler(object):
    def __init__(self):
        pass

    @classmethod
    def from_any(cls, arg):
        if isinstance(arg, Filler):
            return arg
        elif isinstance(arg, (int, float)):
            return ConstantFiller(arg)
        elif isinstance(arg, np.ndarray):
            return CopyFiller(arg)
        elif isinstance(arg, tuple):
            if len(arg) == 2:
                if arg[0] == 'normal':
                    return NormalFiller(**arg[1])
                elif arg[0] == 'uniform':
                    return UniformFiller(**arg[1])
        raise ValueError('Invalid fillter arguments')

    def array(self, shape):
        raise NotImplementedError()


class ConstantFiller(Filler):
    def __init__(self, value=0.0):
        super(ConstantFiller, self).__init__()
        self.value = value

    def array(self, shape):
        return np.ones(shape)*self.value


class NormalFiller(Filler):
    def __init__(self, mu=0.0, sigma=1.0):
        super(NormalFiller, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def array(self, shape):
        array = np.random.normal(loc=self.mu, scale=self.sigma, size=shape)
        return np.array(array)


class UniformFiller(Filler):
    def __init__(self, low=0.0, high=1.0):
        super(UniformFiller, self).__init__()
        self.low = low
        self.high = high

    def array(self, shape):
        array = np.random.uniform(low=self.low, high=self.high, size=shape)
        return np.array(array)


class CopyFiller(Filler):
    def __init__(self, np_array):
        super(CopyFiller, self).__init__()
        self.arr = np_array

    def array(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        if self.arr.shape != shape:
            raise ValueError('Shape mismatch: expected %s but got %s'
                             % (str(self.arr.shape), str(shape)))
        return np.array(self.arr)


class AutoFiller(Filler):
    def __init__(self, gain=1.0):
        super(AutoFiller, self).__init__()
        self.gain = gain

    def array(self, shape):
        ndim = len(shape)
        if ndim == 2:
            # Affine weights
            scale = 1.0 / np.sqrt(shape[0])
        elif ndim == 4:
            # Convolution filter
            scale = 1.0 / np.sqrt(np.prod(shape[1:]))
        else:
            raise ValueError('AutoFiller does not support ndim %i' % ndim)
        scale = self.gain * scale / np.sqrt(3)
        array = np.random.uniform(low=-scale, high=scale, size=shape)
        return np.array(array)


class OrthogonalFiller(Filler):
    def __init__(self, gain=1.0):
        super(OrthogonalFiller, self).__init__()
        self.gain = gain

    def array(self, shape):
        # Implementation inspired by Lasagne.
        # https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
        flat_shape = (shape[0], np.prod(shape[1:]))
        array = np.random.normal(size=flat_shape)
        u, _, v = np.linalg.svd(array, full_matrices=False)
        array = u if u.shape == flat_shape else v
        array = np.reshape(array*self.gain, shape)
        return np.array(array)
