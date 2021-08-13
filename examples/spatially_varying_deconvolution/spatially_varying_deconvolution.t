W,H,Kd,Kc = Dims("W","H","Kd","Kc")
Inputs {
	sqrt_l1 = Param(float, 0),
	sqrt_l2 = Param(float, 1),
	X 	= Unknown(float,{W,H},2),
	M 	= Array(float,{W,H},3),
	b_1 = Array(float,{W,H},4),
	b_2 = Array(float,{W,H},5),
	b_3 = Array(float,{W,H},6),
	K 	= Array(float,{Kd,Kd,Kc},7),
    S   = Sparse({W,H},{Kc},8)
}

k_0=Kd()
k_1=Kd()
x=W()
y=H()
--x,y,k_0,k_1 = W(), H(), Kd(), Kd()
c = S(x,y)

k_half = 8
kx = Sum({k_0,k_1}, K(k_0,k_1,c)*X(x-k_0+k_half, y-k_1+k_half))

Dxx = X(x, y) - X(x-1, y)
Dyx = X(x, y) - X(x, y-1)

E_conv = sqrt_l1 * ( (M(x,y) * kx) - b_1(x,y))
E_dx = sqrt_l2 * (Select(InBounds(x-1),Dxx,0) - b_2(x,y))
E_dy = sqrt_l2 * (Select(InBounds(y-1),Dyx,0) - b_3(x,y))

r = Residuals {
	conv=E_conv,
	dx=E_dx,
	dy=E_dy
}

r.conv.Jp:set_materialize(true)