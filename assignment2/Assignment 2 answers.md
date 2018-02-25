Q1) e)
Automatic differentiation, 

We only need to compute the forward pass,

Q2b)

Exactly 2n: One op/symbol to shift it on to the stack (which 
can only happen left to right, monotonically) and one pop symbol 
to remove it (creating an arc which always removes a symbol).

Q2 f)
\gamma = 1  / (1 - p_drop) for the Expectation under P_drop to be equal to it if there's no drop out.

e.g. p_drop = 0  (no dropout), multiple by 1
e.g. p_drop = 0.999  (all dropped out), multiple by ~ infinity
e.g. p_drop = 0.5  (half dropped out), multiple by 2 to get same output activity

g) i) Like a ball rolling downhill in a narrow valley, the momementum will push it past any minor dips (local) minima.
 It smooths out the movement. By computing the rolling average of gradients and using that, we're 
 effectively computing the gradient of a larger minibatch, which will be closers  to the true gradients.
 
 ii ) Adam keeps a rolling average of the magnitude of the gradients in each dimension. It divides 
 the update by sqrt(v) and so those dims with large gradients will get reduced, and small ones will 
 increase, helping them to escape flat areas/saddle points. Combine this with momentum and you're rolling.