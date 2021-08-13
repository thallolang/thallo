N,K = Dims("N","K")
Inputs {
	C = Unknown(float,{K},0),
	R = Array(float,{N},1),
	T = Array(float,{N}, 2)
}
n,k = N(),K()

local result = Sum({k},R(n-k+2)*C(k))
local e_fit = T(n)-result

e_fit = Select(InBoundsExpanded(n,2), e_fit, 0.0)

r = Residuals {conv = e_fit}

r.conv.Jp:set_materialize(true)