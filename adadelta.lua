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
      print('no state table, ADADELTA initializing')
   end
   local config = config or {}
   local state = state or config
   local wd = state.weightDecay or 0
   local mom = state.momentum or 0
   local damp = state.dampening or mom
   local nesterov = config.nesterov or false
   local rho = state.decay  or 0.95
   local eps = state.epsilon or 1e-6
   local scale = state.scale or 1
   state.evalCounter = state.evalCounter or 0
   assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   -- (2) weight decay
   if wd ~= 0 then
      dfdx:add(wd, x)
   end

   -- (3) apply momentum
   if mom ~= 0 then
      if not state.dfdx then
         state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
      else
         state.dfdx:mul(mom):add(1-damp, dfdx)
      end
      if nesterov then
         dfdx:add(mom, state.dfdx)
      else
         dfdx = state.dfdx
      end
   end

   -- (4) allocate memory for intermediates
   if not state.dfdx_var then
      state.dfdx_var = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):zero()
      state.dx_var = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):zero()
      state.dfdx_var_e = torch.Tensor():typeAs(dfdx):resizeAs(dfdx)
      state.dx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx)
   end

   -- (5) accumulate gradient E[g^2]
   state.dfdx_var:mul(rho):addcmul(1-rho,dfdx,dfdx)

   -- (6) compute update dx
   state.dfdx_var_e:copy(state.dfdx_var):add(eps)
   state.dx:copy(state.dx_var):add(eps):cdiv(state.dfdx_var_e):sqrt():cmul(-dfdx):mul(scale)

   -- (7) accumulate updates E[dx^2]
   state.dx_var:mul(rho):addcmul(1-rho,state.dx,state.dx)

   -- (8) apply update (x = x + dx)
   x:add(state.dx)

   -- (9) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}
end
