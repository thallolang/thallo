N,U = Dims("N","U")
Inputs {
	funcParams =   Unknown(thallo_float2, {U}, 0),
	data =         Image(thallo_float2, {N}, 1)
}
UsePreconditioner(true)

local n = N()
local u = U()
x,y = data(n)(0),data(n)(1)
a,b = funcParams(u)(0),funcParams(u)(1)
Residuals{ fit = y - (a*cos(b*x) + b*sin(a*x)) }
