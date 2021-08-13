local W,H,D = Dims("W","H","D")
Inputs {
	Offset =      Unknown(thallo_float3,{W,H,D},0),
	Angle = 	    Unknown(thallo_float3,{W,H,D},1),	
	UrShape =     Array(thallo_float3,{W,H,D},2), --original position: vertex.xyz
	Constraints = Array(thallo_float3,{W,H,D},3), --user constraints
	w_fitSqrt =   Param(float, 4),
	w_regSqrt =   Param(float, 5)
}
UsePreconditioner(true)
local w,h,d = W(),H(),D()
--fitting
local e_fit = Offset(w,h,d) - Constraints(w,h,d)
local valid = greatereq(Constraints(w,h,d), -999999.9)
reg = {}
for i,j,k in Stencil { {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}} do
    local ow,oh,od = w+i,h+j,d+k
	local ARAPCost = (Offset(w,h,d) - Offset(ow,oh,od)) - Rotate3D(Angle(w,h,d),UrShape(w,h,d) - UrShape(ow,oh,od))
	local ARAPCostF = Select(InBounds(w,h,d),	Select(InBounds(ow,oh,od), ARAPCost, 0.0), 0.0)
	reg[#reg+1] = w_regSqrt*ARAPCostF
end
r = Residuals {
	fit = Select(valid,w_fitSqrt*e_fit,0),
	reg = reg
}