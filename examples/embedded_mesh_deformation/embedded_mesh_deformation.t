local N,E = Dim("N",0),Dim("E",1)
Inputs {
	w_fitSqrt   = Param(float, 0),
	w_regSqrt   = Param(float, 1),
	w_rotSqrt   = Param(float, 2),
	Offset      = Unknown(thallo_float3,{N},3), --vertex.xyz, 
	RotMatrix   = Unknown(thallo_mat3f,{N},4),
	UrShape     = Array(thallo_float3,{N},5),	--urshape: vertex.xyz
	Constraints = Array(thallo_float3,{N},6),	--constraints
	v0          = Sparse({E}, {N}, 7),
	v1          = Sparse({E}, {N}, 8)
}
UsePreconditioner(true)	
n,e = N(),E()

--fitting
local e_fit = Offset(n) - Constraints(n)
local valid = greatereq(Constraints(n)(0), -999999.9)

local regCost = (Offset(v1(e)) - Offset(v0(e))) - 
                gemv(RotMatrix(v0(e)), (UrShape(v1(e)) - UrShape(v0(e))))

--rot
local R = RotMatrix(n)
local c0 = Vector(R(0), R(3), R(6))
local c1 = Vector(R(1), R(4), R(7))
local c2 = Vector(R(2), R(5), R(8))

r = Residuals {
	fit = Select(valid, w_fitSqrt*e_fit, 0),
	reg = w_regSqrt*regCost,
	rot = {
		w_rotSqrt*dot(c0,c1),
		w_rotSqrt*dot(c0,c2),
		w_rotSqrt*dot(c1,c2),
		w_rotSqrt*(dot(c0,c0)-1),
		w_rotSqrt*(dot(c1,c1)-1),
		w_rotSqrt*(dot(c2,c2)-1)
	}
}



