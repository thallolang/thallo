local N,E = Dims("N","E")
Inputs {
	X = Unknown(float,{N},0),
	A = Array(float,{N},1),
	v0 = Sparse({E}, {N}, 2),
	v1 = Sparse({E}, {N}, 3)
}
local n,e = N(), E()

local function weird(x)
    return sin(x)
end

local weirdest = weird(X(n))

local val = weirdest:get(v0(e)) - weirdest:get(v1(e))
--local val = weird(X(v0(e))) - weird(X(v1(e)))
val = val:get(e)
r = Residuals {
	fit = X(n) - A(n), --fitting
	reg = val--regularization
}
