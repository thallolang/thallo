local W,H = Dims("W","H")
Inputs {
    w_fitSqrt         = Param(float, 0),
    w_regSqrtAlbedo   = Param(float, 1),
    w_regSqrtShading  = Param(float, 2),
    pNorm             = Param(thallo_float, 3),
    r                 = Unknown(thallo_float3,{W,H},4),
    i                 = Array(thallo_float3,{W,H},5),
    s                 = Unknown(thallo_float,{W,H},6)
}
local x,y = W(),H()

albedo_reg = {}
-- reg Albedo
for dx,dy in Stencil { {1,0}, {-1,0}, {0,1}, {0, -1} } do
    local ox,oy = x+dx,y+dy
    local diff = (r(x,y) - r(ox,oy))
    -- The helper L_p function takes diff, raises it's length to the (p-2) power, 
    -- makes that expression constant then multiplies it with diff and returns
    local laplacianCost = L_p(diff, pNorm, {x,y})
    local laplacianCostF = Select(InBounds(x,y),Select(InBounds(ox,oy), laplacianCost,0),0)
    albedo_reg[#albedo_reg+1] = w_regSqrtAlbedo*laplacianCostF
end

shading_reg = {}
-- reg Shading
for dx,dy in Stencil { {1,0}, {-1,0}, {0,1}, {0, -1} } do
    local ox,oy = x+dx,y+dy
    local diff = (s(x,y) - s(ox,oy))
    local laplacianCostF = Select(InBounds(x,y),Select(InBounds(ox,oy), diff,0),0)
    shading_reg[#shading_reg+1] = w_regSqrtShading*laplacianCostF
end

-- fit
local fittingCost = r(x,y)+s(x,y)-i(x,y)
res = Residuals {
    fit = w_fitSqrt*fittingCost,
    albedo_reg = albedo_reg,
    shading_reg = shading_reg
}