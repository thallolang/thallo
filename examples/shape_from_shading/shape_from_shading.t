local DEPTH_DISCONTINUITY_THRE = 0.01
local W,H 	= Dims("W","H")
Inputs {
    w_p	      = Param(float,0),-- Fitting weight
    w_s	      = Param(float,1),-- Regularization weight
    w_g	      = Param(float,2),-- Shading weight
    f_x	      = Param(float,3),
    f_y	      = Param(float,4),
    u_x 	  = Param(float,5),
    u_y 	  = Param(float,6),
    L_1       = Param(float,7),
    L_2       = Param(float,8),
    L_3       = Param(float,9),
    L_4       = Param(float,10),
    L_5       = Param(float,11),
    L_6       = Param(float,12),
    L_7       = Param(float,13),
    L_8       = Param(float,14),
    L_9       = Param(float,15),
    X 	      = Unknown(thallo_float, {W,H},16), -- Refined Depth
    D_i 	  = Array(thallo_float, {W,H},17), -- Depth input
    Im 	      = Array(thallo_float, {W,H},18), -- Target Intensity
    edgeMaskR = Array(uint8, {W,H},19), -- Edge mask. 
    edgeMaskC = Array(uint8, {W,H},20), -- Edge mask. 
}

w_p,w_s,w_g = sqrt(w_p), sqrt(w_s), sqrt(w_g)

local x,y = W(), H()
local posX,posY = x:asvalue(),y:asvalue()

-- equation 8
function p(offX,offY) 
    local d = X(x+offX,y+offY)
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

    local n_x = X(_x, _y - 1) * (X(_x, _y) - X(_x - 1, _y)) / f_y
    local n_y = X(_x - 1, _y) * (X(_x, _y) - X(_x, _y - 1)) / f_x
    local n_z = (n_x * (u_x - i) / f_x) + (n_y * (u_y - j) / f_y) - (X(_x-1, _y)*X(_x, _y-1) / (f_x*f_y))
    local sqLength = n_x*n_x + n_y*n_y + n_z*n_z
    local inverseMagnitude = Select(greater(sqLength, 0.0), 1.0/sqrt(sqLength), 1.0)
    return inverseMagnitude * Vector(n_x, n_y, n_z)
end

function B(offX, offY)
	local normal = normalAt(offX, offY)
	local n_x = normal[0]
	local n_y = normal[1]
	local n_z = normal[2]

	return  L_1 +
			L_2*n_y + L_3*n_z + L_4*n_x  +
			L_5*n_x*n_y + L_6*n_y*n_z + L_7*(-n_x*n_x - n_y*n_y + 2*n_z*n_z) + L_8*n_z*n_x + L_9*(n_x*n_x-n_y*n_y)
end

function I(offX, offY)
	return Im(x+offX,y+offY)*0.5 + 0.25*(Im(x+offX-1,y+offY)+Im(x+offX,y+offY-1))
end

local function DepthValid(offX,offY) return greater(D_i(x+offX,y+offY),0) end
 
local function B_I(offX,offY)
    local bi = B(offX,offY) - I(offX,offY)
    local valid = DepthValid(offX-1,offY)*DepthValid(offX,offY)*DepthValid(offX,offY-1)
    return Select(valid,bi,0) -- Select(InBoundsExpanded(x,y,1)*valid,bi,0)
end

B_I_comp = B_I(0,0)
function B_I(offX,offY) return B_I_comp:get(x+offX,y+offY) end

-- do not include unknowns for where the depth is invalid
--X:Exclude(Not(DepthValid(0,0)))

-- fitting term
local E_p = X(x,y) - D_i(x,y)
E_p = Select(DepthValid(0,0),w_p*E_p,0)

-- shading term
local E_g_h = (B_I(0,0) - B_I(1,0))*edgeMaskR(x,y)
local E_g_v = (B_I(0,0) - B_I(0,1))*edgeMaskC(x,y)
E_g_h = Select(InBoundsExpanded(x,y,1),w_g*E_g_h,0)
E_g_v = Select(InBoundsExpanded(x,y,1),w_g*E_g_v,0)

-- regularization term
local function Continuous(offX,offY) return less(abs(X(x,y) - X(x+offX,y+offY)),DEPTH_DISCONTINUITY_THRE) end

local valid = DepthValid(0,0)*DepthValid(0,-1)*DepthValid(0,1)*DepthValid(-1,0)*DepthValid(1,0)*
                  Continuous(0,-1)*Continuous(0,1)*Continuous(-1,0)*Continuous(1,0)--*InBoundsExpanded(x,y,1)
valid = eq(valid:get(x,y),1)

local E_s = 4.0*p(0,0) - (p(-1,0) + p(0,-1) + p(1,0) + p(0,1)) 
E_s = Select(valid,w_s*E_s,0)

r = Residuals {
    fit = E_p,
    shading_h = E_g_h,
    shading_v = E_g_v,
    reg = E_s
}
--local merged = r:merge(r:merge(r.fit,r.shading),r.reg):compute_at_output(true)
--merged.JtF:compute_at_output(true)
--B_I_comp:materialize(true),valid:materialize(true)