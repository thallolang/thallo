local N,M,U = Dims("N", "M", "U")
Inputs {
	BlendshapeWeights = Unknown(thallo_float, {M}, 0),
	AverageMesh       = Array(thallo_float3, {N}, 1),
	BlendshapeBasis   = Array(thallo_float3, {N,M}, 2),
	Target            = Array(thallo_float2, {N}, 4),
	w_regSqrt         = Param(float, 5),
	CamParams         = Array(thallo_float9, {U}, 6), -- Workaround for scalar-only parameters
}  
UsePreconditioner(true)

function snavely_projection(point, params)
	-- params[0,1,2] are the axis-angle rotation
	p = AngleAxisRotatePoint(params:slice(0,3), point)
	-- params[3,4,5] are the translation.
	p = p + params:slice(3,6)

	-- Compute the center of distortion. The sign change comes from
	-- the camera model that Noah Snavely's Bundler assumes, whereby
	-- the camera coordinate system has a negative z axis.
	center_of_distortion = Vector(-p(0) / p(2), -p(1) / p(2))

	-- Apply second and fourth order radial distortion.
	l1 = params(7)
	l2 = params(8)
	r2 = dot(center_of_distortion,center_of_distortion)
	distortion = 1.0 + r2 * (l1 + l2  * r2)

	-- Compute final projected point position.
	focal = params(6)
	projected = center_of_distortion * focal * distortion
	return projected
end

local m,n,u = M(),N(),U()
local camera = CamParams(u)
local Mesh = AverageMesh(n) + Sum({m}, BlendshapeBasis(n,m) * BlendshapeWeights(m))
local Pos2D = snavely_projection(Mesh, camera)
local e_fit = Target(n) - Pos2D
local valid = greatereq(Target(n,0), -999999.9)
r = Residuals {
	reg = w_regSqrt*BlendshapeWeights(m),
	fit = Select(valid,e_fit,0)
}
r.fit.J:set_materialize(true)