local N,E = Dims("N","E")
Inputs {
	X = Unknown(float3,{N},0),
	A = Array(float3,{N},1),
	v0 = Sparse({E}, {N}, 2),
	v1 = Sparse({E}, {N}, 3)
}
local n,e = N(), E()

local function weird(x)
    return Vector(x, x, x, x, x, x, x, x, x, x, x, x)
end

local weirdest = weird(sin(X(n)(0)+X(n)(1)+X(n)(2)))

local val = weirdest:get(v0(e)) - weirdest:get(v1(e))
--local val = weird(X(v0(e))) - weird(X(v1(e)))
local final_val = val
final_val = val:get(e)
r = Residuals {
	fit = X(n) - A(n), --fitting
	reg = final_val--regularization
}
weirdest:set_gradient_materialize(false)
val:set_gradient_materialize(true)
weirdest:set_materialize(true)
val:set_materialize(false)