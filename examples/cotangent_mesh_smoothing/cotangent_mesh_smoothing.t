N,E = Dims("N","E")
Inputs {
	w_fitSqrt = Param(float, 0),
	w_regSqrt = Param(float, 1),
	X = Unknown(thallo_float3,{N},2),
	A = Array(thallo_float3,{N},3),
	V0 = Sparse({E}, {N}, 4), --current vertex
	V1 = Sparse({E}, {N}, 5), --neighboring vertex
	V2 = Sparse({E}, {N}, 6), --prev neighboring vertex
	V3 = Sparse({E}, {N}, 7)  --next neighboring vertex
}
UsePreconditioner(true)

function cot(p0, p1) 
	local adotb = dot(p0, p1)
	local disc = dot(p0, p0)*dot(p1, p1) - adotb*adotb
	disc = Select(greater(disc, 0.0), disc,  0.0001)
	return dot(p0, p1) / Sqrt(disc)
end

local n = N()
local e = E()
local v0,v1,v2,v3 = V0(e),V1(e),V2(e),V3(e)


local a = normalize(X(v0) - X(v2)) --float3
local b = normalize(X(v1) - X(v2)) --float3
local c = normalize(X(v0) - X(v3)) --float3
local d = normalize(X(v1) - X(v3)) --float3

--cotangent laplacian; Meyer et al. 03
local w = 0.5*(cot(a,b) + cot(c,d))
w = Sqrt(Select(greater(w, 0.0), w, 0.0001))

r = Residuals {
	fit = w_fitSqrt*(X(n) - A(n)),
	reg = w_regSqrt*w*(X(v1) - X(v0))
}

