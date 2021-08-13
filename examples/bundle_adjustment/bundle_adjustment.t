C, P, O = Dims("C", "P", "O")
Inputs {
    cameras = Unknown(thallo_float9,{ C }, 0),
    points = Unknown(thallo_float3, { P }, 1),
    observations = Array(float2, { O }, 2),
    oToC = Sparse({ O }, { C }, 3),
    oToP = Sparse({ O }, { P }, 4)
}
UsePreconditioner(true)
local o = O()
local camera, point = cameras(oToC(o)), points(oToP(o))
-- camera[0,1,2] are the axis-angle rotation
p = AngleAxisRotatePoint(camera:slice(0,3), point)
-- camera[3,4,5] are the translation.
p = p + camera:slice(3,6)

-- Compute the center of distortion. The sign change comes from
-- the camera model that Noah Snavely's Bundler assumes, whereby
-- the camera coordinate system has a negative z axis.
center_of_distortion = Vector(-p(0) / p(2), -p(1) / p(2))

-- Apply second and fourth order radial distortion.
l1 = camera(7)
l2 = camera(8)
r2 = dot(center_of_distortion,center_of_distortion)
distortion = 1.0 + r2  * (l1 + l2  * r2)

-- Compute final projected point position.
focal = camera(6)
predicted = center_of_distortion * focal * distortion
observed = observations(o)

--oToC:set_coherent(true)
r = Residuals { snavely_reprojection_error = observed-predicted }
