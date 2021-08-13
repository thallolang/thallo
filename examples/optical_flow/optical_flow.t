local W,H = Dims("W","H")
Inputs {
	w_fitSqrt = Param(float, 0),
	w_regSqrt = Param(float, 1),
	X         = Unknown(thallo_float2,{W,H},2),
	I         = Array(thallo_float,{W,H},3),
	I_hat_im  = Array(thallo_float,{W,H},4),
	I_hat_dx  = Array(thallo_float,{W,H},5),
	I_hat_dy  = Array(thallo_float,{W,H},6)
}
local I_hat = SampledImage(I_hat_im,I_hat_dx,I_hat_dy)


local x = W()
local y = H()
local i,j = x:asvalue(), y:asvalue()
UsePreconditioner(false)
-- fitting
local e_fit = w_fitSqrt*(I(x,y) - I_hat(i + X(x,y)(0),j + X(x,y)(1)))

reg = {}
-- regularization
for ox,oy in Stencil { {1,0}, {-1,0}, {0,1}, {0,-1} } do
    local nx,ny = x+ox,y+oy
	local e_reg = w_regSqrt*(X(x,y) - X(nx,ny))
    reg[#reg+1] = Select(InBounds(nx,ny),e_reg,0)
end

r = Residuals {
	fit = e_fit,
	reg_px = reg[1],
    reg_nx = reg[2],
    reg_py = reg[3],
    reg_ny = reg[4]
}
--r:merge(r.fit,r.reg):compute_at_output(true)
