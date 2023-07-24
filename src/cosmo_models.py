import numpy as np
import astropy.cosmology as cosmo

from base_models import Model

class Cosmology(Model):

    def __init__(
        self,
    ):
        
        super().__init__()
    
    def distance_modulus(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class FlatLambdaCDM(Cosmology):

    def __init__(
        self,
        H0: float = 70.0,
        Om0: float = 0.3,
    ):
        
        super().__init__()
        self.H0 = H0
        self.Om0 = Om0
        self.cosmo = cosmo.FlatLambdaCDM(H0=self.H0, Om0=self.Om0)
    
    def distance_modulus(self, z: np.ndarray) -> np.ndarray:
        return self.cosmo.distmod(z).value

class FlatwCDM(Cosmology):

    def __init__(
        self,
        H0: float = 70.0,
        Om0: float = 0.3,
        w0: float = -1.0,
    ):
        
        super().__init__()
        self.H0 = H0
        self.Om0 = Om0
        self.w0 = w0
        self.cosmo = cosmo.FlatwCDM(H0=self.H0, Om0=self.Om0, w0=self.w0)
    
    def distance_modulus(self, z: np.ndarray) -> np.ndarray:
        return self.cosmo.distmod(z).value

class Flatw0waCDM(Cosmology):

    def __init__(
        self,
        H0: float = 70.0,
        Om0: float = 0.3,
        w0: float = -1.0,
        wa: float = 0.0,
    ):
        
        super().__init__()
        self.H0 = H0
        self.Om0 = Om0
        self.w0 = w0
        self.wa = wa
        self.cosmo = cosmo.Flatw0waCDM(H0=self.H0, Om0=self.Om0, w0=self.w0, wa=self.wa)
    
    def distance_modulus(self, z: np.ndarray) -> np.ndarray:
        return self.cosmo.distmod(z).value