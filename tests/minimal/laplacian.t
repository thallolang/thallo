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
		Select(InBounds(x+1,y+1),(X(x,y) - X(x+1,y)),0),
		Select(InBounds(x,y+1),(X(x,y) - X(x,y+1)),0)
	}
}

r.fit.J:set_materialize(true)
r.fit.JtJ:set_materialize(true)

r.reg.J:set_materialize(true)
r.reg.JtJ:set_materialize(true)
