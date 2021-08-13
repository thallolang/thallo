N,E = Dims("N","E")
Inputs {
	w_fitSqrt = Param(float, 0),
	w_regSqrt = Param(float, 1),
	Position = Unknown(thallo_float3,{N},2),
	Angle = Unknown(thallo_float3,{N},3),	
	Original = Array(thallo_float3,{N},4),
	Constraints = Array(thallo_float3,{N},5), --user constraints
	V0 = Sparse({E}, {N}, 6),
	V1 = Sparse({E}, {N}, 7)
}

UsePreconditioner(true)
n,e = N(),E()
v0,v1 = V0(e),V1(e)
local e_fit = Position(n) - Constraints(n)
local valid = greatereq(Constraints(n)(0), -999999.9)
local ARAPCost = (Position(v0) - Position(v1)) - Rotate3D(Angle(v0),Original(v0) - Original(v1))
r = Residuals {
	fit = Select(valid,w_fitSqrt*e_fit,0),
	reg = w_regSqrt*ARAPCost
}
-- Default schedule gives good performance