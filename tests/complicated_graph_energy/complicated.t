local X,E = Dims("X", "E")
Inputs {
	U =   Unknown(float2, {X}, 0),
	Cor =	Array(float2, {X}, 1)
	A = Sparse({E}, {X}, 3)
	B = Sparse({E}, {X}, 4)
}
x,e = X(), E()

local C = Cor(A(e))
local UA = U(A(e))
local UB = U(B(e))

local wA = UA(0) * C(0) + C(0)
local wB = UB(0) * C(1) + UB(1)

r = Residuals {
	r0 = wA - wB
	r1 = U(x) * Cor(x)
}
