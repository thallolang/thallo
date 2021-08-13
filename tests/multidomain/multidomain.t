N,U = Dims("N","U")
Inputs {
	offset = Unknown(thallo_float, {U}, 0),
	pts    = Unknown(thallo_float, {N}, 1),
    target = Array(thallo_float, {N}, 2)
}
n,u = N(),U()
x,x_0 = pts(n), target(n)
Residuals{ 
    fit = offset(u) + x - x_0,
    reg = x
}
