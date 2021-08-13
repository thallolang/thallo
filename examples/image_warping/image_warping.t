local W,H = Dims("W","H")
Inputs {
	Offset = Unknown(thallo_float2,{W,H},0),
	Angle = Unknown(thallo_float,{W,H},1),	
	UrShape = Array(thallo_float2,{W,H},2), --original mesh position
	Constraints = Array(thallo_float2,{W,H},3), -- user constraints
	Mask = Array(thallo_float, {W,H},4), -- validity mask for mesh
	w_fitSqrt = Param(float, 5),
	w_regSqrt = Param(float, 6)
}
UsePreconditioner(true)
local x = W()
local y = H()
Offset:Exclude(Not(eq(Mask(x,y),0)))
Angle:Exclude(Not(eq(Mask(x,y),0)))

local regs = {}
for dx,dy in Stencil { {1,0}, {-1,0}, {0,1}, {0, -1} } do
    local e_reg = w_regSqrt*((Offset(x,y) - Offset(x+dx,y+dy)) 
                             - Rotate2D(Angle(x,y),(UrShape(x,y) - UrShape(x+dx,y+dy))))
    local valid = InBounds(x+dx,y+dy) * eq(Mask(x,y),0) * eq(Mask(x+dx,y+dy),0)
    regs[#regs+1] = Select(valid,e_reg,0)
end
local e_fit = (Offset(x,y) - Constraints(x,y))
local valid = All(greatereq(Constraints(x,y),0))*eq(Mask(x,y),0)
r = Residuals {
    reg_px = regs[1],
    reg_nx = regs[2],
    reg_py = regs[3],
    reg_ny = regs[4],
    fit = w_fitSqrt*Select(valid, e_fit , 0.0)
}
--local merged = r:merge(r.reg,r.fit):compute_at_output(true)
--merged.JtF:compute_at_output(true)