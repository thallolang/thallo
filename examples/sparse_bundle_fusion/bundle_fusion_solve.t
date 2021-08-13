local W,H,T,CorrDim = Dims("W","H","T","CorrDim")
Inputs {
	CamTranslation = Unknown(thallo_float3,{T},0),
	CamRotation    = Unknown(thallo_float3,{T},1),
	Pos_j = Array(	float3, {CorrDim},2),
	Pos_i = Array(	float3, {CorrDim},3),
	weightSparse = Param(float, 4),
	corr_i       = Sparse({CorrDim}, {T}, 5),
	corr_j       = Sparse({CorrDim}, {T}, 6)
}

--UsePreconditioner(true)

local t = T()
local transform_t = function(t) return PoseToMatrix(CamRotation(t), CamTranslation(t)) end

-- Sparse Residual
local c = CorrDim()
local i,j = corr_i(c),corr_j(c)

TI = transform_t(i)
TJ = transform_t(j)

TI = transform_t(t):get(i)

r = rigid_trans(TI,Pos_i(c)) - rigid_trans(TJ,Pos_j(c));
res = ad.Vector(r(0), r(1), r(2))

r = Residuals { 
	sparse = Sqrt(weightSparse)*res
}

corr_i:set_coherent(true)
corr_j:set_coherent(true)

--r.sparse.JtJ:set_materialize(true)
--r.sparse.Jp:set_materialize(true)