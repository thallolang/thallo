W,H = Dims("W","H")
Inputs {
	X = Unknown(float,{W,H},0),
	A = Array(float,{W,H},1)
}
x,y = W(),H()

v = X(x,y) - X(x+1,y)
v = Select(InBounds(x+1,y), v, 0.0)
local reg = v:get(x,y)
--local reg = v - (X(x,y+1) - X(x+1,y+1))

r = Residuals {
	fit = (X(x,y) - A(x,y)),
	reg = reg
}