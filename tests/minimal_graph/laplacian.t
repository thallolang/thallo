local N,E = Dims("N","E")
Inputs {
	X = Unknown(float,{N},0),
	A = Array(float,{N},1),
	v0 = Sparse({E}, {N}, 2),
	v1 = Sparse({E}, {N}, 3)
}
w_fit = .5

n,e = N(), E()
r = Residuals {
	fit = w_fit*(X(n) - A(n)), --fitting
	reg = X(v0(e)) - X(v1(e))--regularization
}
--[[
r.reg.J:set_materialize(true)
r.reg.JtJ:set_materialize(true)
--]]
r.fit.J:set_materialize(true)
r.fit.JtJ:set_materialize(true)
