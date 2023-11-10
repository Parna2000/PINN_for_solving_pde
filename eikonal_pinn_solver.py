import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pinn_solver import PINNSolver

class EikonalPINNSolver(PINNSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def fun_r(self, t, x, u, u_t, u_x, u_xx):
        """Residual of the PDE"""
        return -u_t + tf.abs(u_x) - 1.
    
    def get_r(self):
        """We update get_r since the Eikonal equation is a first-order equation.
        Therefore, it is not necessary to compute second derivatives."""
        
        with tf.GradientTape(persistent=True) as tape:
            # Watch variables representing t and x during this GradientTape
            tape.watch(self.t)
            tape.watch(self.x)
            
            # Compute current values u(t,x)
            u = self.model(tf.stack([self.t[:,0], self.x[:,0]], axis=1))
            
        u_x = tape.gradient(u, self.x)
            
        u_t = tape.gradient(u, self.t)
        
        del tape
        
        return self.fun_r(self.t, self.x, u, u_t, u_x, None)