
# PINN_for_solving_pde

Partial differential equations are ubiquitous in mathematically oriented scientific fields, such as physics and engineering. For instance, they are foundational in the modern scientific knowledge of sound, heat, diffusion, electrostatics, electrodynamics, thermodynamics, fluid dynamics, elasticity, general relativity, and quantum mechanics (Schrödinger equation, Pauli equation, etc.). They also arise from many purely mathematical considerations, such as differential geometry and the calculus of variations; among other notable applications, they are the fundamental tool in the proof of the Poincaré conjecture from geometric topology. 
But not all PDEs can be solved by analytical methods and this is a major limitation we are facing nowadays. One such problem is the Eikonal equation. Eikonal equations naturally arise in the WKB method and the study of Maxwell's equations. Eikonal equations provide a link between physical (wave) optics and geometric (ray) optics.
So, here we will use another very rapidly developing field of study, i.e., Deep Learning. It is growing at an exponential rate and getting used in almost every field of study. Its major applications include automatic speech recognition, image recognition, visual art processing, natural language processing, etc. Nowadays, it is also used for solving partial differential equations.
Here, we will solve the Eikonal equation by using Neural Networks. More precisely, we will be using Physics Informed Neural Networks (PINNs), and Using physics-informed neural networks does not require the often expensive mesh generation that conventional CFD methods rely on. So, here, we will not only be solving the Eikonal equation but we will also be getting a method by which we will be able to solve almost all differential equations with minor changes in the codes and this will help us a lot in different fields of study.



## Acknowledgements

 - [PINN for pdes](https://www.sciencedirect.com/science/article/abs/pii/S0045782522001438#:~:text=PINNs%20embed%20the%20PDE%20residual,even%20with%20many%20training%20points.)
 - [A Physics Informed Neural Network for Time-Dependent Nonlinear and Higher Order Partial Differential Equations](https://arxiv.org/abs/2106.07606)
 - [Introduction to Physics-informed Neural Networks](https://towardsdatascience.com/solving-differential-equations-with-neural-networks-afdcf7b8bcc4)


## Deployment
The Ekonal equation, that we are solving, is:\
![](https://github.com/Parna2000/PINN_for_solving_pde/blob/main/pictures/Picture2.png?raw=true)
### Import necessary packages and set problem specific data
This code runs with TensorFlow version 2.10.0. The implementation relies mainly on the scientific computing library NumPy and the machine learning library TensorFlow.
All computations were performed on an Intel(R) Core(TM) i3-1005G1 CPU @ 1.20GHz   1.19 GHz within a couple of minutes.


```bash
import tensorflow as tf
import numpy as nf

# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

# Define initial condition
def Eikonal_u_0(x):
    n = x.shape[0]
    return tf.zeros((n,1), dtype=DTYPE)

# Define boundary condition
def Eikonal_u_b(t, x):
    n = x.shape[0]
    return tf.zeros((n,1), dtype=DTYPE)

```
### Generate a set of collocation points
We assume that the starting time and boundary data points X0 and Xb, as well as the collocation points Xr, are produced by random sampling from a uniform distribution. We'll sample consistently throughout this. We select equally distributed beginning value and boundary points Nb=50 and N0=50, and we sample Nr=10000 collocation points evenly throughout the domain boundaries.

```bash
N_0 = 50
N_b = 50
N_r = 10000

tmin = 0.
tmax = 1.
xmin = -1.
xmax = 1.

lb = tf.constant([tmin, xmin], dtype=DTYPE)
ub = tf.constant([tmax, xmax], dtype=DTYPE)

tf.random.set_seed(0)

t_0 = tf.ones((N_0,1), dtype=DTYPE)*lb[0]
x_0 = tf.random.uniform((N_0,1), lb[1], ub[1], dtype=DTYPE)
X_0 = tf.concat([t_0, x_0], axis=1)

u_0 = Eikonal_u_0(x_0)

t_b = tf.random.uniform((N_b,1), lb[0], ub[0], dtype=DTYPE)
x_b = lb[1] + (ub[1] - lb[1]) * t1.keras.backend.random_bernoulli((N_b,1), 0.5, dtype=DTYPE)
X_b = tf.concat([t_b, x_b], axis=1)

u_b = Eikonal_u_b(t_b, x_b)

t_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
x_r = tf.random.uniform((N_r,1), lb[1], ub[1], dtype=DTYPE)
X_r = tf.concat([t_r, x_r], axis=1)

X_data = [X_0, X_b]
u_data = [u_0, u_b]

The positions where the boundary and beginning conditions will be applied are then shown, along with the collocation points (red circles)(cross marks, color indicates value).
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9,6))
plt.scatter(t_0, x_0, c=u_0, marker='X', vmin=-1, vmax=1)
plt.scatter(t_b, x_b, c=u_b, marker='X', vmin=-1, vmax=1)
plt.scatter(t_r, x_r, c='r', marker='.', alpha=0.1)
plt.xlabel('$t$')
plt.ylabel('$x$')

plt.title('Positions of collocation points and boundary data');
#plt.savefig('Xdata_Burgers.pdf', bbox_inches='tight', dpi=300)

```
This generates the following plot:\
![](https://github.com/Parna2000/PINN_for_solving_pde/blob/main/pictures/Picture3.png?raw=true)
### General network architecture
Here, we assume a feedforward neural network of the following structure:\
•	the input is scaled elementwise to lie in the interval [−1,1],\
•	followed by 8 fully connected layers each containing 20 neurons and each followed by a hyperbolic tangent activation function,\
•	one fully connected output layer.\
A network with 3021 trainable parameters is produced by this setup (first hidden layer: 2.20+20=60; seven intermediate layers: each 20.20+20=420; output layer: 20.1+1=21). In order to leverage this architecture for more testing, we will build it up in a class called "PINN NeuralNet()."

```bash
class PINN_NeuralNet(tf.keras.Model):
    """ Set basic architecture of the PINN model."""

    def __init__(self, lb, ub, 
            output_dim=1,
            num_hidden_layers=8, 
            num_neurons_per_layer=20,
            activation='tanh',
            kernel_initializer='glorot_normal',
            **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.lb = lb
        self.ub = ub
        
        # Define NN architecture
        self.scale = tf.keras.layers.Lambda(
            lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(output_dim)
        
    def call(self, X):
        """Forward-pass through neural network."""
        Z = self.scale(X)
        for i in range(self.num_hidden_layers):
            Z = self.hidden[i](Z)
        return self.out(Z)

```
### Construct base class to solve any PDE
Here we will construct the class PINNSolver()
```bash
import scipy.optimize

class PINNSolver():
    def __init__(self, model, X_r):
        self.model = model
        
        self.t = X_r[:,0:1]
        self.x = X_r[:,1:2]
        
        self.hist = []
        self.iter = 0
    
    def get_r(self):
        
            with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.t)
            tape.watch(self.x)
            
            u = self.model(tf.stack([self.t[:,0], self.x[:,0]], axis=1))

            u_x = tape.gradient(u, self.x)
            
        u_t = tape.gradient(u, self.t)
        u_xx = tape.gradient(u_x, self.x)
        
        del tape
        
        return self.fun_r(self.t, self.x, u, u_t, u_x, u_xx)
    
    def loss_fn(self, X, u):
        
        r = self.get_r()
        phi_r = tf.reduce_mean(tf.square(r))
        
        
        loss = phi_r

        
        for i in range(len(X)):
            u_pred = self.model(X[i])
            loss += tf.reduce_mean(tf.square(u[i] - u_pred))
        
        return loss
    
    def get_grad(self, X, u):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.loss_fn(X, u)
            
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        
        return loss, g
    
   def solve_with_TFoptimizer(self, optimizer, X, u, N=1001):
        
        @tf.function
        def train_step():
            loss, grad_theta = self.get_grad(X, u)
            
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss
        
        for i in range(N):
            
            loss = train_step()
            
            self.current_loss = loss.numpy()
            self.callback()

```
### Derive Eikonal solver class
From the PINNSolver() class, we now derive a solver for the Eikonal equation. We build a new method called get_r() that avoids computing second derivatives because the Eikonal equation does not depend on second-order derivatives.
```bash
class EikonalPINNSolver(PINNSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def fun_r(self, t, x, u, u_t, u_x, u_xx):

        return -u_t + tf.abs(u_x) - 1.
    
    def get_r(self):
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.t)
            tape.watch(self.x)
            
            u = self.model(tf.stack([self.t[:,0], self.x[:,0]], axis=1))
            
        u_x = tape.gradient(u, self.x)
            
        u_t = tape.gradient(u, self.t)
        
        del tape
        
        return self.fun_r(self.t, self.x, u, u_t, u_x, None)

```
### Neural network architecture for Eikonal equation
The neural network model chosen for this particular problem can be simpler. We decided to use only two hidden layers with 20 neurons in each, resulting in 501 unknown parameters (first hidden layer: 2⋅20+20=60; one intermediate layer: 20⋅20+20=420; output layer: 20⋅1+1=21). To account for the lack of smoothness of the solution, we choose a non-differentiable activation function, although the hyperbolic tangent function seems to be able to approximate the kinks in the solution sufficiently well. Here, we decided to use the leaky rectified linear unit (leaky ReLU) activation function
![](https://github.com/Parna2000/PINN_for_solving_pde/blob/main/pictures/Picture4.png?raw=true)
which displays a non-vanishing gradient when the unit is not active, i.e., when z<0.

```bash
model = PINN_NeuralNet(lb, ub, num_hidden_layers=2,
                       activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                       kernel_initializer='he_normal')

model.build(input_shape=(None,2))
eikonalSolver = EikonalPINNSolver(model, X_r)

```

### Training the model

Here, the optimizer is configured and the model is trained. To do this, we set the learning rate to the step function after initialising the model.

![](https://github.com/Parna2000/PINN_for_solving_pde/blob/main/pictures/Picture5.png?raw=true)

Then we set up a tf.keras.optimizer to train the model, and which decays in a piecewise constant manner. Here, N=10001 epochs are used to train the model.
It will take about 40 seconds to do this.

```bash
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([3000,7000],[1e-1,1e-2,1e-3])

optim = tf.keras.optimizers.Adam(learning_rate=lr)
t0 = time()

eikonalSolver.solve_with_TFoptimizer(optim, X_data, u_data, N=10001)

print('\nComputation time: {} seconds'.format(time()-t0))

```
The first 10 iterations will look like:

It 00000: loss = 4.59581718e+01\
It 00050: loss = 8.44158530e-02\
It 00100: loss = 5.00384606e-02\
It 00150: loss = 2.98081823e-02\
It 00200: loss = 1.63681470e-02\
It 00250: loss = 9.09537822e-03\
It 00300: loss = 7.32776895e-03\
It 00350: loss = 6.97098300e-03\
It 00400: loss = 5.94768114e-03\
It 00450: loss = 4.51994361e-03

### Plotting the results
```
eikonalSolver.plot_solution();
```

This will give the following 3D plot:

![App Screenshot](https://github.com/Parna2000/PINN_for_solving_pde/blob/main/pictures/Picture1.png?raw=true)

