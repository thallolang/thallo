W,H = Dims("W","H")
Inputs {
	X = Unknown(float,{W,H},0),
	A = Array(float,{W,H},1),
    Xn = Sparse({W,H},{W},2),
    Yn = Sparse({W,H},{H},3)
}
w_fit = .2
x,y = W(), H()
print(x)
xn = Xn(x,y)
yn = Yn(x,y)
r = Residuals {
	fit = w_fit*(X(x,y) - A(x,y)),
	reg = {
		(X(x,y) - X(xn,y)),
		(X(x,y) - X(x,yn))
	}
}
