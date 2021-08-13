N,M = Dims("N","M")
Inputs {
	W = Unknown(float,{M},0),
	S = Array(float,{N,M},1),
	T = Array(float,{N}, 2)
}
n,m = N(),M()

local result = Sum({m},S(n,m)*W(m))

r = Residuals { fit = T(n)-result }
r.fit.Jp:set_materialize()