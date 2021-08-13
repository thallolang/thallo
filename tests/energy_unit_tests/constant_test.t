N,U,E = Dims("N","U","E")
Inputs {
	funcParams = Unknown(thallo_float2, {U}, 0),
	data       = Array(thallo_float2, {N}, 1),
	D          = Sparse({E}, {N}, 2),
	P          = Sparse({E}, {U}, 3)
}
--UsePreconditioner(true)
e,n,u = E(),N(),U()

local x = function(n) return data(n)(0) end
local a = function(u) return funcParams(u)(0) end

x = x(D(e))
a = a(u):get(P(e))

y = data(D(e))(1)
b = funcParams(P(e))(1)
r = Residuals { sparse = Constant(b)*y - b*x }

r.fit.JtJp:set_materialize()
