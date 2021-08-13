-- Switch to double to check for precision issues in the solver
-- using double incurs bandwidth, compute, and atomic performance penalties
if _thallo_double_precision then
	thallo_float =  double
else
	thallo_float =  float
end