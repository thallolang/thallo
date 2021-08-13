#pragma once

#include "../../shared/cudaUtil.h"

#include "WarpingSolverUtil.h"
#include "WarpingSolverState.h"
#include "WarpingSolverParameters.h"
#include "RotationHelper.h"

////////////////////////////////////////
// evalF
////////////////////////////////////////

inline __host__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

// Adapted From Ceres
__device__ float3 AngleAxisRotatePoint(const float3 axis_angle, const float3 pt) {
    float3 result;
    const float theta2 = dot(axis_angle, axis_angle);
    // -- std::numeric_limits<double>::epsilon() in Ceres, we use our own cutoff here
    if (theta2 >  1e-8) {
        // Away from zero, use the rodriguez formula
        //
        //   result = pt costheta +
        //            (w x pt) * sintheta +
        //            w (w . pt) (1 - costheta)
        //
        // We want to be careful to only evaluate the square root if the
        // norm of the angle_axis vector is greater than zero. Otherwise
        // we get a division by zero.
        //
        const float theta = sqrtf(theta2);
        const float costheta = cosf(theta);
        const float sintheta = sinf(theta);
        const float theta_inverse = 1.0 / theta;

        const float3 w = axis_angle * theta_inverse;

        // Explicitly inlined evaluation of the cross product for
        // performance reasons.

        const float3 w_cross_pt = cross(w, pt);
        const float tmp =
           dot(w,pt) * (1.0f - costheta);
        result = pt*costheta + w_cross_pt*sintheta + w * tmp;
    }
    else {
        // Near zero, the first order Taylor approximation of the rotation
        // matrix R corresponding to a vector w and angle w is
        //
        //   R = I + hat(w) * sin(theta)
        //
        // But sintheta ~ theta and theta * w = angle_axis, which gives us
        //
        //  R = I + hat(w)
        //
        // and actually performing multiplication with the point pt, gives us
        // R * pt = pt + w x pt.
        //
        // Switching to the Taylor expansion near zero provides meaningful
        // derivatives when evaluated using Jets.
        const float3 w_cross_pt = cross(axis_angle, pt);

        result = pt + w_cross_pt;
    }
    return result;
}

float2 __device__ snavelyProjection(const float* const camera,
    float3 point) {
    // camera[0,1,2] are the angle-axis rotation.
    
    float3 axis_angle = make_float3(camera[0], camera[1], camera[2]);
    float3 p = AngleAxisRotatePoint(axis_angle, point);

    // camera[3,4,5] are the translation.
    p += make_float3(camera[3], camera[4], camera[5]);

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    float xp = -p.x / p.z;
    float yp = -p.y / p.z;
    float r2 = xp*xp + yp*yp;
    // Apply second and fourth order radial distortion.
    const float l1 = camera[7];
    const float l2 = camera[8];
    
    float distortion = 1.0f + r2  * (l1 + l2  * r2);
    const float focal = camera[6];

    return make_float2(focal * distortion * xp, focal * distortion * yp);
}


float2 __device__ snavelyDerivatives(const float* const camera,
    float3 point, float3 bb) {
    // camera[0,1,2] are the angle-axis rotation.

    float3 axis_angle = make_float3(camera[0], camera[1], camera[2]);
    float3 p = AngleAxisRotatePoint(axis_angle, point);

    // camera[3,4,5] are the translation.
    p += make_float3(camera[3], camera[4], camera[5]);

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    float xp = -p.x / p.z;
    float yp = -p.y / p.z;
    float r2 = xp*xp + yp*yp;
    // Apply second and fourth order radial distortion.
    const float l1 = camera[7];
    const float l2 = camera[8];

    float distortion = 1.0f + r2  * (l1 + l2  * r2);
    const float focal = camera[6];

    float3 v = AngleAxisRotatePoint(axis_angle, bb);

    float coeff = (l1 + l2*r2) + l2*r2;
    float tmp = 2 * coeff*(v.x*xp + v.y*yp + v.z*r2);
    float partialx = (focal / p.z)*(distortion*v.x + xp*distortion*v.z + xp*tmp);
    float partialy = (focal / p.z)*(distortion*v.y + yp*distortion*v.z + yp*tmp);

    return make_float2(partialx, partialy);
}


__inline__ __device__ float evalFDevice(unsigned int vertexIdx, SolverInput& input, SolverState& state, SolverParameters& parameters) {
    float2 e = make_float2(0.0f, 0.0f);
	// E_fit
	if (state.d_target[vertexIdx].x != MINF) {
		float3 v = state.d_mesh[vertexIdx];
		float2 t = state.d_target[vertexIdx];

        float2 e_fit = t - snavelyProjection(state.d_camParams, v);
		e += e_fit*e_fit;
	}
	float res = e.x + e.y;
	// E_reg
	if (vertexIdx < input.M) {
		float e_reg = sqrt(input.wReg)*state.d_blendshapeWeights[vertexIdx];
		res += e_reg*e_reg;
	}
	return res;
}