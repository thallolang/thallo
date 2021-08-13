W,H = Dims("W","H")
Inputs {
	X = Unknown(float,{W,H},0),
	A = Array(float,{W,H},1)
}
w_fit = .2
x,y = W(), H()
r = Residuals {
	fit = w_fit*(X(x,y) - A(x,y)),
	reg = {
		(X(x,y) - X(x+1,y)),
		(X(x,y) - X(x,y+1))
	}
}
