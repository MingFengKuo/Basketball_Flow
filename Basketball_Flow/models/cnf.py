import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_normal


__all__ = ["CNF", "SequentialFlow"]


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows."""
    
    def __init__(self, layer_list):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layer_list)
        
    def forward(self, x, context, logpx=None, reverse=False, inds=None, integration_times=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, context, logpx, integration_times, reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, context, logpx, integration_times, reverse)
            return x, logpx

class CNF(nn.Module):
    def __init__(self, odefunc, conditional=True, T=1.0, train_T=False, regularization_fns=None,
                 solver='dopri5', atol=1e-5, rtol=1e-5, use_adjoint=True):
        super(CNF, self).__init__()
        self.train_T = train_T
        self.T = T
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
            print("Training T :", self.T)
        
        if regularization_fns is not None and len(regularization_fns) > 0:
            raise NotImplementedError("Regularization not supported")
            
        self.use_adjoint = use_adjoint
        self.odefunc = odefunc
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}
        self.adjoint_options = {"norm": "seminorm"}
        self.conditional = conditional
        
    def forward(self, x, context=None, logpx=None, integration_times=None, reverse=False):
        if logpx is None:
            _logpx = torch.zeros(*x.shape[:-1], 1).to(x)
        else:
            _logpx = logpx
        
        if self.conditional:
            assert context is not None
            states = (x, _logpx, context)
        else:
            states = (x, _logpx)

        atol = self.atol
        rtol = self.rtol
            
        if integration_times is None:
            if self.train_T:
                integration_times = torch.stack(
                    [torch.zeros(()).to(x), self.sqrt_end_time * self.sqrt_end_time]
                ).to(x)
            else:
                integration_times = torch.tensor([0., self.T], requires_grad=False).to(x)
    
        if reverse:
            integration_times = _flip(integration_times, 0)
        
        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()
        if self.use_adjoint:
            state_t = odeint_adjoint(
                self.odefunc,
                states,
                integration_times.to(x),
                atol=atol,
                rtol=rtol,
                method=self.solver,
                options=self.solver_options,
                adjoint_options=self.adjoint_options
        )
        else:
            state_t = odeint_normal(
                self.odefunc,
                states,
                integration_times.to(x),
                atol=atol,
                rtol=rtol,
                method=self.solver,
                options=self.solver_options,
            )
            
        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)
            
        z_t, logpz_t = state_t[:2]
        
        if logpx is not None:
            return z_t, logpz_t
        else:
            return z_t
    
    def num_evals(self):
        return self.odefunc._num_evals.item()
        
def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]