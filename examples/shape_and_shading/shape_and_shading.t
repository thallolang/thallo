local DEPTH_DISCONTINUITY_THRE = 0.01
local W,H,U 	= Dims("W","H","U")
Inputs {
    w_p	      = Param(float,0),-- Fitting weight
    w_s	      = Param(float,1),-- Regularization weight
    w_g	      = Param(float,2),-- Shading weight
    f_x	      = Param(float,3),
    f_y	      = Param(float,4),
    u_x 	  = Param(float,5),
    u_y 	  = Param(float,6),
    ell       = Unknown(thallo_float9,{U},7),
    D_r 	  = Unknown(thallo_float, {W,H},8), -- Refined Depth
    D_i 	  = Array(thallo_float, {W,H},9), -- Depth input
    Im 	      = Array(thallo_float, {W,H},10), -- Target Intensity
    edgeMaskR = Array(uint8, {W,H},11), -- Edge mask. 
    edgeMaskC = Array(uint8, {W,H},12), -- Edge mask. 
}

w_p,w_s,w_g = sqrt(w_p), sqrt(w_s), sqrt(w_g)

local x,y,u = W(), H(),U()
local posX,posY = x:asvalue(),y:asvalue()

-- equation 8
function p(offX,offY) 
    local d = D_r(x+offX,y+offY)
    local i = offX + posX
    local j = offY + posY
    return Vector(((i-u_x)/f_x)*d, ((j-u_y)/f_y)*d, d)
end

-- equation 10
function normalAt(offX, offY)
    local i = offX + posX -- good
    local j = offY + posY -- good
    
    local _x = x+offX
    local _y = y+offY

    local n_x = D_r(_x, _y - 1) * (D_r(_x, _y) - D_r(_x - 1, _y)) / f_y
    local n_y = D_r(_x - 1, _y) * (D_r(_x, _y) - D_r(_x, _y - 1)) / f_x
    local n_z = (n_x * (u_x - i) / f_x) + (n_y * (u_y - j) / f_y) - (D_r(_x-1, _y)*D_r(_x, _y-1) / (f_x*f_y))
    local sqLength = n_x*n_x + n_y*n_y + n_z*n_z
    local inverseMagnitude = Select(greater(sqLength, 0.0), 1.0/sqrt(sqLength), 1.0)
    return inverseMagnitude * Vector(n_x, n_y, n_z)
end


normExp = normalAt(0,0)

function B(offX, offY)
	local n = normExp:get(x+offX, y+offY) --TODO: why does wrapping this in a function give improper results?
	local n_x = n[0]
	local n_y = n[1]
	local n_z = n[2]
	return  ell(u)(0) +
			ell(u)(1)*n_y + ell(u)(2)*n_z + ell(u)(3)*n_x  +
			ell(u)(4)*n_x*n_y + ell(u)(5)*n_y*n_z + ell(u)(6)*(-n_x*n_x - n_y*n_y + 2*n_z*n_z) + ell(u)(7)*n_z*n_x + ell(u)(8)*(n_x*n_x-n_y*n_y)
end

function I(offX, offY)
	return Im(x+offX,y+offY)*0.5 + 0.25*(Im(x+offX-1,y+offY)+Im(x+offX,y+offY-1))
end

local function DepthValid(offX,offY) return greater(D_i(x+offX,y+offY),0) end
 
local function B_I(offX,offY)
    local bi = B(offX,offY) - I(offX,offY)
    local valid = DepthValid(offX-1,offY)*DepthValid(offX,offY)*DepthValid(offX,offY-1)
    return Select(valid,bi,0)
end

B_I_comp = B_I(0,0)

-- fitting term
local E_p = D_r(x,y) - D_i(x,y)
E_p = Select(DepthValid(0,0),w_p*E_p,0)

-- shading term
local E_g_h = Select(eq(edgeMaskR(x,y),1),(B_I(0,0) - B_I(1,0)),0)
local E_g_v = Select(eq(edgeMaskC(x,y),1),(B_I(0,0) - B_I(0,1)),0)
E_g_h = w_g*E_g_h
E_g_v = w_g*E_g_v

-- regularization term
local function Continuous(offX,offY) return less(abs(D_r(x,y) - D_r(x+offX,y+offY)),DEPTH_DISCONTINUITY_THRE) end

local valid = DepthValid(0,0)*DepthValid(0,-1)*DepthValid(0,1)*DepthValid(-1,0)*DepthValid(1,0)*
                  Continuous(0,-1)*Continuous(0,1)*Continuous(-1,0)*Continuous(1,0)

valid = eq(valid:get(x,y),1)

local E_s = 4.0*p(0,0) - (p(-1,0) + p(0,-1) + p(1,0) + p(0,1)) 
E_s = Select(valid,w_s*E_s,0)
E_lighting = Select(valid,0.1*B_I(0,0),0)
r = Residuals {
    fit = E_p,
    shading_grad = {E_g_h,E_g_v},
    lighting = E_lighting,
    reg = E_s
}
--local merged = r:merge(r:merge(r.fit,r.shading),r.reg)
--B_I_comp:materialize(true),valid:materialize(true)
--local L_merged = r:merge(r.shading_grad,r.lighting)
--L_merged:reorder({u,x,y})
--B_I_comp:set_materialize(false)
--r.lighting.J:set_materialize(true)
--r.shading_grad.J:set_materialize(true)
--B_I_comp:set_materialize(false)
--B_I_comp:set_gradient_materialize(false)
--normExp:set_gradient_materialize(false)
r:merge(r.shading_grad,r.lighting)
r:merge(r.fit,r.reg)