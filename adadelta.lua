--[[ ADADELTA implementation of SGD

ARGS:
- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- `x`      : the initial point
- `state`  : a table describing the state of the optimizer; after each
             call the state is modified
- `state.decay`   : decay constant of exponential moving average
- `state.epsilon` : for numerical stability
- `state.scale`   : scale parameter updates

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

]]
function optim.adadelta(opfunc, x, config, state)
   -- (0) get/update state
   if config == nil and state == nil then
      print('no state table, ADAGRAD initializing')
   end
   local config = config or {}
   local state = state or config
   local rho = state.decay  or 0.95
   local eps = state.epsilon or 1e-6
   local scale = state.scale or 1
   state.evalCounter = state.evalCounter or 0

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   -- (2) allocate memory for intermediates
   if not state.dfdx_var then
      state.dfdx_var = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):zero()
      state.dx_var = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):zero()
   end

   -- (4) accumulate gradient E[g^2]
   state.dfdx_var:mul(rho):addcmul(1-rho,dfdx,dfdx)

   -- (5) compute update dx
   local dfdx_std = torch.add(state.dfdx_var,eps):sqrt()
   local dx_std = torch.add(state.dx_var,eps):sqrt()
   local dx = dx_std:cdiv(dfdx_std):cmul(-dfdx):mul(scale)

   -- (6) accumulate updates E[dx^2]
   state.dx_var:mul(rho):addcmul(1-rho,dx,dx)

   -- (7) apply update (x = x + dx)
   x:add(dx)

   -- (8) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}
end
