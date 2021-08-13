local N,E = Dims("N","E")
Inputs {
	w_fitSqrt         = Param(float, 0),
	w_regSqrt         = Param(float, 1),
	Offset            = Unknown(thallo_float3,{N},2),
	Angle             = Unknown(thallo_float3,{N},3),
	RobustWeights     = Unknown(thallo_float,{N},4),	
	UrShape           = Array(thallo_float3, {N},5),        --urshape: vertex.xyz
	Constraints       = Array(thallo_float3,{N},6),	    --constraints
	ConstraintNormals = Array(thallo_float3,{N},7),
	v0				  = Sparse({E}, {N}, 8),
	v1				  = Sparse({E}, {N}, 9)
}
w_confSqrt        = 0.1
UsePreconditioner(true)

local n = N()
local e = E()

local robustWeight = RobustWeights(n)
--fitting
local e_fit = robustWeight*ConstraintNormals(n):dot(Offset(n) - Constraints(n))
local validConstraint = greatereq(Constraints(n), -999999.9)


--RobustWeight Penalty
local e_conf = 1-(robustWeight*robustWeight)
e_conf = Select(validConstraint, e_conf, 0.0)


--regularization
local ARAPCost = (Offset(v0(e)) - Offset(v1(e))) - Rotate3D(Angle(v0(e)),UrShape(v0(e)) - UrShape(v1(e)))

r = Residuals {
    fit = w_fitSqrt*Select(validConstraint, e_fit, 0.0),
    conf = w_confSqrt*e_conf,
    reg = w_regSqrt*ARAPCost
}