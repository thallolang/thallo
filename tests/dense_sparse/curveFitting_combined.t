N,U,E = Dims("N","U","E")
Inputs {
	funcParams = Unknown(thallo_float2, {U}, 0),
	data       = Array(thallo_float2, {N}, 1),
	D          = Sparse({E}, {N}, 2),
	P          = Sparse({E}, {U}, 3)
}
UsePreconditioner(true)

e,n,u = E(),N(),U()

y = data(n)(1)
b = funcParams(u)(1)
aFun = function(id) 
    local sqA = funcParams(id)(0)
    return sqrt(sqA)
end
local x = function(id) 
    local sqrtX = data(id)(0)
    return sqrtX*sqrtX
end

local xx = x(n):get(n)
local aa = aFun(u):get(u)
--Energy(

y = data(D(e))(1)
b = funcParams(P(e))(1)
local xx = x(D(e))
local aa = a(P(e))

Reesiduals {
	dense = y - (aa*cos(b*xx) + b*sin(aa*xx)),
	sparse = y - (aa*cos(b*xx) + b*sin(aa*xx))
}