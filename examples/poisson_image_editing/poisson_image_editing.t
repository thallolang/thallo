local W,H = Dims("W","H")
Inputs {
	X = Unknown(thallo_float4, { W, H }, 0), --unknown, initialized to base image
	T = Array(thallo_float4,{W,H},1), -- inserted image
	M = Array(thallo_float, {W,H},2) -- mask, excludes parts of base image
}
UsePreconditioner(false)
local x,y = W(),H()
X:Exclude(neq(M(x,y),0))
reg = {}
for dx,dy in Stencil { {1,0}, {-1,0}, {0,1}, {0, -1} } do
    local ox,oy = x+dx,y+dy
    local e = (X(x,y) - X(ox,oy)) - (T(x,y) - T(ox,oy))
    reg[#reg+1] = Select(InBounds(ox, oy),Select(eq(M(x,y),0), e, 0),0)
end
r = Residuals {
    reg_px = reg[1],
    reg_nx = reg[2],
    reg_py = reg[3],
    reg_ny = reg[4]
 }
--r.reg:compute_at_output(true)
--r.reg.JtF:compute_at_output(true)