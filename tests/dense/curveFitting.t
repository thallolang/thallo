N,U,E = Dims("N","U","E")
Inputs {
	funcParams = Unknown(thallo_float2, {U}, 0),
	data       = Array(thallo_float2, {N}, 1),
	D          = Sparse({E}, {N}, 2),
	P          = Sparse({E}, {U}, 3)
}
UsePreconditioner(true)

local e = E()
x,y = data(D(e))(0),data(D(e))(1)
a,b = funcParams(P(e))(0),funcParams(P(e))(1)
Residuals{ fit = y - (a*cos(b*x) + b*sin(a*x)) }
