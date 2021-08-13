local W,H,T,CorrDim,PairDim = Dims("W","H","T","CorrDim","PairDim")
Inputs {
	CamTranslation = Unknown(thallo_float3,{T},0),
	CamRotation    = Unknown(thallo_float3,{T},1),
	ConstCamTranslation   = Array(thallo_float3,{T},0),
	ConstCamRotation      = Array(thallo_float3,{T},1),
	Positions = Array(   float4, {W,H,T},2),
	Normals   = Array(     float4, {W,H,T},3),
	Pos_j = Array(   float3, {CorrDim},4),
	Pos_i = Array(   float3, {CorrDim},5),
	depthMin      = Param(float, 6),
	depthMax      = Param(float, 7),
	normalThresh  = Param(float, 8),
	distThresh    = Param(float, 9),
	fx = Param(float, 10),
	fy = Param(float, 11),
	cx = Param(float, 12),
	cy = Param(float, 13),
	imageWidth            = Param(float, 14),
	imageHeight           = Param(float, 15),
	weightDenseDepth      = Param(float, 16),
	weightSparse          = Param(float, 17),
	corr_i                = Sparse({CorrDim}, {T}, 18),
	corr_j                = Sparse({CorrDim}, {T}, 19),
	t_target              = Sparse({CorrDim}, {T}, 20),
	t_source              = Sparse({CorrDim}, {T}, 21)
}
local InterpolatedPositions = SampledImageArray(Positions)
local InterpolatedNormals   = SampledImageArray(Normals)

-- Dense Depth Residual
w,h,p = W(), H(), PairDim()
t_s, t_t = t_source(p),t_target(p)

local camPosSrc = Positions(w,h,t_s)
local nrmj  = Normals(w,h,t_s)

local validSrcPos = greater(camPosSrc(2),depthMin)*less(camPosSrc(2),depthMax)
local validSrcNormal = neq(nrmj(0), -inf)

local t0,t1 = T(),T()
local transform_t = function(t) return PoseToMatrix(CamRotation(t0), CamTranslation(t0)):get(t) end
local invtransform_t = function(t) return InvertRigidTransform(transform_t(t)) end
local consttransform_t = function(t) return PoseToMatrix(ConstCamRotation(t), ConstCamTranslation(t)) end
local constinvtransform_t = function(t) return InvertRigidTransform(consttransform_t(t)) end

function GetTransform(transform,invtransform, i_index, j_index)
    local transform_j = transform(j_index)
    local inv_transform_i = invtransform(i_index)
    return Mat4ToRigidTransform(matmul(inv_transform_i,transform_j))
end
function NonConstGetTransform(i_index, j_index)
    return GetTransform(transform_t, constinvtransform_t, i_index, j_index)
end

local transform = NonConstGetTransform(t0,t1):get(t_t, t_s)
nrmj = Vec3(gemv(transform,nrmj))

camPosSrcToTgt = rigid_trans(transform,camPosSrc)
tgtScreenPosf = CameraToDepth(fx, fy, cx, cy, Constant(camPosSrcToTgt))
local inScreen =    greatereq(tgtScreenPosf(0), -0.5) * greatereq(tgtScreenPosf(1), -0.5) * 
                    less(tgtScreenPosf(0), (imageWidth + 0.5)) * less(tgtScreenPosf(1), (imageHeight + 0.5))

cposi = InterpolatedPositions(tgtScreenPosf(0),tgtScreenPosf(1), t_t:asvalue())
local validTgtPos = greater(cposi(2),depthMin) * less(cposi(2), depthMax)
nrmi = Vec3(InterpolatedNormals(tgtScreenPosf(0),tgtScreenPosf(1), t_t:asvalue()))
local validTgtNormal = neq(nrmi(0), -inf)
camPosTgt = Vec3(cposi)

dist = length(camPosSrcToTgt,camPosTgt)
dNormal = dot(nrmj, nrmi)
local closeEnough = greatereq(dNormal, normalThresh) * lesseq(dist,distThresh)

diff = camPosTgt - camPosSrcToTgt
depthRes = dot(diff, nrmi)

depthRes = SelectOnAll({validSrcPos,validSrcNormal,inScreen,validTgtPos,validTgtNormal,closeEnough},depthRes,0.0)

imPairWeight = 1.0
depthWeight = weightDenseDepth * imPairWeight * (pow(Max(0.0, 1.0 - camPosTgt(2) / 2.0), 2.5))

-- Sparse Residual
local c = CorrDim()
i,j = corr_i(c),corr_j(c)
r = rigid_trans(transform_t(i),Pos_i(c)) - rigid_trans(transform_t(j),Pos_j(c));
res = Vector(r(0), r(1), r(2))
r = Residuals {
    dense = Sqrt(depthWeight)*depthRes,
    sparse = Sqrt(weightSparse)*res
}

r.dense.JtJ:set_materialize(true)
r.sparse.JtJ:set_materialize(true)
r.dense:reorder({w,h,p})