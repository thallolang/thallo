local N,U = Dims("N","U")
Inputs {
	Translation = Unknown(thallo_float3,{U},0), 
	Angle       = Unknown(thallo_float3,{U},1),
	Mesh        = Array(thallo_float3,{N},2),    --original position
	Target      = Array(thallo_float3,{N},3)
}
UsePreconditioner(true)
n,u = N(),U()
local valid = greatereq(Target(n)(0), -999999.9)
E_fit = Select(valid,Rotate3D(Angle(u),Mesh(n)) + Translation(u) - Target(n),0)
r = Residuals { fit = E_fit }
r.fit.JtJ:set_materialize(true)