
class Model():

    def __init__(self) -> None:
        pass

    def log_likelihood(self, **kwargs):
        
        raise NotImplementedError

class Gaussian(Model):

    def __init__(self) -> None:
        super().__init__()

    def residual(self, **kwargs):
        
        raise NotImplementedError
    
    def covariance(self, **kwargs):
        
        raise NotImplementedError
    
class Mixture(Model):

    def __init__(
        self,
    ) -> None:
        super().__init__()
        