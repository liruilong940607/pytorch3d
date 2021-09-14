/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_CALC_GRADIENTS_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_CALC_GRADIENTS_H_

#include <math.h> 
#include "../global.h"
#include "./commands.h"
#include "./renderer.h"

#include "./renderer.draw.device.h"

namespace pulsar {
namespace Renderer {

template <bool DEV>
GLOBAL void calc_gradients_sampling_nerf(
    const CamInfo cam, /** Camera in world coordinates. */
    float const* const RESTRICT grad_im, /** The gradient image. */
    const float
        gamma, /** The transparency parameter used in the forward pass. */
    float3 const* const RESTRICT vert_poss, /** Vertex position vector. */
    float const* const RESTRICT vert_cols, /** Vertex color vector. */
    float const* const RESTRICT vert_rads, /** Vertex radius vector. */
    float const* const RESTRICT opacity, /** Vertex opacity. */
    float const* const RESTRICT bg_col, /** bg colors. */
    const uint num_balls, /** Number of balls. */
    float const* const RESTRICT result_d, /** Result image. */
    float const* const RESTRICT forw_info_d, /** Forward pass info. */
    DrawInfo const* const RESTRICT di_d, /** Draw information. */
    IntersectInfo const* const RESTRICT ii_d, /** Intersect information. */
    // Mode switches.
    const bool calc_grad_pos,
    const bool calc_grad_col,
    const bool calc_grad_rad,
    const bool calc_grad_cam,
    const bool calc_grad_opy,
    // Out variables.
    float* const RESTRICT grad_rad_d, /** Radius gradients. */
    float* const RESTRICT grad_col_d, /** Color gradients. */
    float3* const RESTRICT grad_pos_d, /** Position gradients. */
    CamGradInfo* const RESTRICT grad_cam_buf_d, /** Camera gradient buffer. */
    float* const RESTRICT grad_opy_d, /** Opacity gradient buffer. */
    int* const RESTRICT
        grad_contributed_d, /** Gradient contribution counter. */
    // Infrastructure.
    const int n_track,
    const uint offs_x,
    const uint offs_y /** Debug offsets. */
) {
  uint limit_x = cam.film_width, limit_y = cam.film_height;
  if (offs_x != 0) {
    // We're in debug mode.
    limit_x = 1;
    limit_y = 1;
  }
  GET_PARALLEL_IDS_2D(coord_x_base, coord_y_base, limit_x, limit_y);
  // coord_x_base and coord_y_base are in the film coordinate system.
  // We now need to translate to the aperture coordinate system. If
  // the principal point was shifted left/up nothing has to be
  // subtracted - only shift needs to be added in case it has been
  // shifted down/right.
  const uint film_coord_x = coord_x_base + offs_x;
  const uint ap_coord_x = film_coord_x +
      2 * static_cast<uint>(std::max(0, cam.principal_point_offset_x));
  const uint film_coord_y = coord_y_base + offs_y;
  const uint ap_coord_y = film_coord_y +
      2 * static_cast<uint>(std::max(0, cam.principal_point_offset_y));
  float* result = const_cast<float*>(
      result_d + film_coord_y * cam.film_width * cam.n_channels +
      film_coord_x * cam.n_channels);
  const float* grad_im_l = grad_im +
      film_coord_y * cam.film_width * cam.n_channels +
      film_coord_x * cam.n_channels;
  // Set up shared infrastructure.
  const int fwi_loc = film_coord_y * cam.film_width * (3 + 2 * n_track) +
      film_coord_x * (3 + 2 * n_track);
  float3 ray_dir_norm; /** Ray cast through the pixel, normalized. */
  if (cam.orthogonal_projection) {
    ray_dir_norm = cam.sensor_dir_z;
  } else {
    ray_dir_norm = normalize(
        cam.pixel_0_0_center + film_coord_x * cam.pixel_dir_x +
        film_coord_y * cam.pixel_dir_y);
    // This is a reasonable assumption for normal focal lengths and image sizes.
    PASSERT(FABS(ray_dir_norm.z) > FEPS);
  }
  // Start processing. 
  int n_samples = 64;
  float t_min = cam.min_dist;
  float t_max = cam.max_dist;
  // Both the first and the last samples are included.
  float delta_t = (t_max - t_min) / (n_samples - 1);
  // PASS 1
  float accum = 0.f;
  float light_intensity;
  light_intensity = 1.f;
  for (int sample_id = 0; sample_id < n_samples; ++sample_id) {
    float t = t_min + delta_t * sample_id;
    /** The sampled point in the camera coordinate system. */
    float3 p = ray_dir_norm * t;
    // For each sampled point, we accumulate its attributes by
    // IDW (inverse distance weighted) interpolation across
    // all the spheres that are interated with it.
    float denominator = 0.f;
    float p_sigma = 0.f;
    // TODO(ruilongli): hard-code n_channels to 3 here. Need to dynamiclly
    // construct a vector to hold the values.
    float p_col_ptr[3] = {0.f};
    for (int i = 0; i < n_track; ++i) {
      int sphere_id;
      FASI(forw_info_d[fwi_loc + 3 + 2 * i], sphere_id);
      if (sphere_id == -1) continue;
      // float closest_possible_intersection =
      //   di_d[sphere_id].t_center - di_d[sphere_id].radius;
      // float furthest_possible_intersection =
      //   di_d[sphere_id].t_center + di_d[sphere_id].radius;
      // if (
      //   t < closest_possible_intersection
      //   or t > furthest_possible_intersection
      // ) continue;
      /** The sphere center in the camera coordinate system. */
      float3 center = di_d[sphere_id].ray_center_norm * di_d[sphere_id].t_center;
      /** The distance between the sphere and the sampled point */
      float dist = length(center - p);
      // if (dist > di_d[sphere_id].radius) continue;
      float dist_pow = FMAX(pow(dist, 6), FEPS);
      denominator += 1.0f / dist_pow;
      float sigma = opacity == NULL ? 0.f : opacity[sphere_id];
      PASSERT(isfinite(p_sigma));
      p_sigma += sigma / dist_pow;
      float const* const col_ptr =
        cam.n_channels > 3 ? di_d[sphere_id].color_union.ptr : &di_d[sphere_id].first_color;
      for (uint c_id = 0; c_id < cam.n_channels; ++c_id) {
        PASSERT(isfinite(p_col_ptr[c_id]));
        p_col_ptr[c_id] += col_ptr[c_id] / dist_pow;
      }
    }
    if (denominator > 0) {
      p_sigma /= denominator;
      for (uint c_id = 0; c_id < cam.n_channels; ++c_id) {
        p_col_ptr[c_id] /= denominator;
      }
    } else {
      continue;
    }
    float att;
    // if (sample_id == n_samples - 1) att = 0.f;  // delta_t is MAX_FLOAT
    // else att = exp(- delta_t * p_sigma);
    att = exp(- delta_t * p_sigma);
    float weight = light_intensity * (1.f - att);
    float total_color = 0.f;
    for (uint c_id = 0; c_id < cam.n_channels; ++c_id) {
      total_color += p_col_ptr[c_id] * grad_im_l[c_id];
      for (int i = 0; i < n_track; ++i) {
        int sphere_id;
        FASI(forw_info_d[fwi_loc + 3 + 2 * i], sphere_id);
        if (sphere_id == -1) continue;
        // float closest_possible_intersection =
        //   di_d[sphere_id].t_center - di_d[sphere_id].radius;
        // float furthest_possible_intersection =
        //   di_d[sphere_id].t_center + di_d[sphere_id].radius;
        // if (
        //   t < closest_possible_intersection
        //   or t > furthest_possible_intersection
        // ) continue;
        /** The sphere center in the camera coordinate system. */
        float3 center = di_d[sphere_id].ray_center_norm * di_d[sphere_id].t_center;
        /** The distance between the sphere and the sampled point */
        float dist = length(center - p);
        // if (dist > di_d[sphere_id].radius) continue;
        float dist_pow = FMAX(pow(dist, 6), FEPS);
        ATOMICADD(
          &(grad_col_d[sphere_id * cam.n_channels + c_id]),
          1.0f / dist_pow / denominator * weight * grad_im_l[c_id]); // TODO: be careful about zero.
        if (c_id == 0) {
          PULSAR_LOG_DEV_APIX(
            PULSAR_LOG_GRAD,
            "grad|nerf color. i(%d), sphere_id(%d), t(%.5f), "
            "sigma(%.5f), att(%.5f), weight(%.5f), dist_pow(%.5f), denominator(%.5f), "
            "grad_im(%.5f, %.5f, %.5f) \n",
            i, sphere_id, t, 
            p_sigma, att, weight, dist_pow, denominator,
            grad_im_l[0], grad_im_l[1], grad_im_l[2]);  
        }
      }
    }
    light_intensity *= att;
    accum += weight * total_color;
  }
  // float total_bg = 0.f;
  // for (uint c_id = 0; c_id < cam.n_channels; ++c_id) {
  //   total_bg += bg_col[c_id] * grad_im_l[c_id];
  // }
  // accum += light_intensity * total_bg;
  // // PASS 2
  // light_intensity = 1.f;
  // for (int grad_idx = 0; grad_idx < n_track; ++grad_idx) {
  //   int sphere_idx;
  //   FASI(forw_info_d[fwi_loc + 3 + 2 * grad_idx], sphere_idx);
  //   PASSERT(
  //       sphere_idx == -1 ||
  //       sphere_idx >= 0 && static_cast<uint>(sphere_idx) < num_balls);
  //   float t = forw_info_d[fwi_loc + 3 + 2 * grad_idx + 1]; // sphere depth
  //   if (sphere_idx >= 0 && t >= cam.min_dist) {
  //     float t_next;
  //     if (grad_idx < n_track - 1) {
  //       t_next = 
  //         forw_info_d[fwi_loc + 3 + 2 * (grad_idx + 1) + 1] == -1.f 
  //         ? MAX_FLOAT
  //         : forw_info_d[fwi_loc + 3 + 2 * (grad_idx + 1) + 1];
  //     } else {
  //       t_next = MAX_FLOAT;
  //     }
  //     if (t_next == MAX_FLOAT)
  //       // For the last sphere, the opacity gradient is always zero.
  //       // So we can safely skip it. This is a workaround to avoid 
  //       // numeric error caused by exp(- delta_t * sigma)
  //       break;
  //     float delta_t = FMIN(FABS(t_next - t), 1e10);
  //     float sigma = opacity == NULL ? MAX_FLOAT : opacity[sphere_idx];
  //     // Do not use FEXP() here. The error is not marginal!
  //     float att = exp(- delta_t * sigma);
  //     float weight = light_intensity * (1.f - att);
  //     float const* const col_ptr =
  //       cam.n_channels > 3 ? di_d[sphere_idx].color_union.ptr : &di_d[sphere_idx].first_color;
  //     float total_color = 0.f;
  //     for (uint c_id = 0; c_id < cam.n_channels; ++c_id) {
  //       total_color += col_ptr[c_id] * grad_im_l[c_id];
  //     }
  //     light_intensity *= att;
  //     accum -= weight * total_color;
  //     float grad_opy = delta_t * (total_color * light_intensity - accum);
  //     ATOMICADD(&(grad_opy_d[sphere_idx]), grad_opy);
  //     PULSAR_LOG_DEV_APIX(
  //         PULSAR_LOG_GRAD,
  //         "grad|nerf opacity. grad_idx(%d), sphere_idx(%d), t(%.5f), "
  //         "sigma(%.5f), att(%.5f), alpha(%.5f), "
  //         "T(%.5f), weight(%.5f), total_color(%.5f), "
  //         "accum(%.5f), "
  //         "grad_opy(%.5f), "
  //         "grad_im(%.5f, %.5f, %.5f) \n",
  //         grad_idx, sphere_idx, t, 
  //         sigma, att, 1.f - att,
  //         light_intensity, weight, total_color,
  //         accum,
  //         grad_opy,
  //         grad_im_l[0], grad_im_l[1], grad_im_l[2]);  
  //   }
  // }
  END_PARALLEL_2D_NORET();
};

template <bool DEV>
GLOBAL void calc_gradients_nerf(
    const CamInfo cam, /** Camera in world coordinates. */
    float const* const RESTRICT grad_im, /** The gradient image. */
    const float
        gamma, /** The transparency parameter used in the forward pass. */
    float3 const* const RESTRICT vert_poss, /** Vertex position vector. */
    float const* const RESTRICT vert_cols, /** Vertex color vector. */
    float const* const RESTRICT vert_rads, /** Vertex radius vector. */
    float const* const RESTRICT opacity, /** Vertex opacity. */
    float const* const RESTRICT bg_col, /** bg colors. */
    const uint num_balls, /** Number of balls. */
    float const* const RESTRICT result_d, /** Result image. */
    float const* const RESTRICT forw_info_d, /** Forward pass info. */
    DrawInfo const* const RESTRICT di_d, /** Draw information. */
    IntersectInfo const* const RESTRICT ii_d, /** Intersect information. */
    // Mode switches.
    const bool calc_grad_pos,
    const bool calc_grad_col,
    const bool calc_grad_rad,
    const bool calc_grad_cam,
    const bool calc_grad_opy,
    // Out variables.
    float* const RESTRICT grad_rad_d, /** Radius gradients. */
    float* const RESTRICT grad_col_d, /** Color gradients. */
    float3* const RESTRICT grad_pos_d, /** Position gradients. */
    CamGradInfo* const RESTRICT grad_cam_buf_d, /** Camera gradient buffer. */
    float* const RESTRICT grad_opy_d, /** Opacity gradient buffer. */
    int* const RESTRICT
        grad_contributed_d, /** Gradient contribution counter. */
    // Infrastructure.
    const int n_track,
    const uint offs_x,
    const uint offs_y /** Debug offsets. */
) {
  uint limit_x = cam.film_width, limit_y = cam.film_height;
  if (offs_x != 0) {
    // We're in debug mode.
    limit_x = 1;
    limit_y = 1;
  }
  GET_PARALLEL_IDS_2D(coord_x_base, coord_y_base, limit_x, limit_y);
  // coord_x_base and coord_y_base are in the film coordinate system.
  // We now need to translate to the aperture coordinate system. If
  // the principal point was shifted left/up nothing has to be
  // subtracted - only shift needs to be added in case it has been
  // shifted down/right.
  const uint film_coord_x = coord_x_base + offs_x;
  const uint ap_coord_x = film_coord_x +
      2 * static_cast<uint>(std::max(0, cam.principal_point_offset_x));
  const uint film_coord_y = coord_y_base + offs_y;
  const uint ap_coord_y = film_coord_y +
      2 * static_cast<uint>(std::max(0, cam.principal_point_offset_y));
  float* result = const_cast<float*>(
      result_d + film_coord_y * cam.film_width * cam.n_channels +
      film_coord_x * cam.n_channels);
  const float* grad_im_l = grad_im +
      film_coord_y * cam.film_width * cam.n_channels +
      film_coord_x * cam.n_channels;
  // Set up shared infrastructure.
  const int fwi_loc = film_coord_y * cam.film_width * (3 + 2 * n_track) +
      film_coord_x * (3 + 2 * n_track);
  // Start processing. 
  float accum = 0.f;
  float light_intensity;
  // PASS 1
  light_intensity = 1.f;
  for (int grad_idx = 0; grad_idx < n_track; ++grad_idx) {
    int sphere_idx;
    FASI(forw_info_d[fwi_loc + 3 + 2 * grad_idx], sphere_idx);
    PASSERT(
        sphere_idx == -1 ||
        sphere_idx >= 0 && static_cast<uint>(sphere_idx) < num_balls);
    float t = forw_info_d[fwi_loc + 3 + 2 * grad_idx + 1]; // sphere depth
    if (sphere_idx >= 0 && t >= cam.min_dist) {
      float t_next;
      if (grad_idx < n_track - 1) {
        t_next = 
          forw_info_d[fwi_loc + 3 + 2 * (grad_idx + 1) + 1] == -1.f 
          ? MAX_FLOAT
          : forw_info_d[fwi_loc + 3 + 2 * (grad_idx + 1) + 1];
      } else {
        t_next = MAX_FLOAT;
      }
      float delta_t = FMIN(FABS(t_next - t), 1e10);
      float sigma = opacity == NULL ? MAX_FLOAT : opacity[sphere_idx];
      // Do not use FEXP() here. The error is not marginal!
      float att = exp(- delta_t * sigma);
      float weight = light_intensity * (1.f - att);
      float const* const col_ptr =
        cam.n_channels > 3 ? di_d[sphere_idx].color_union.ptr : &di_d[sphere_idx].first_color;
      float total_color = 0.f;
      for (uint c_id = 0; c_id < cam.n_channels; ++c_id) {
        total_color += col_ptr[c_id] * grad_im_l[c_id];
        ATOMICADD(
            &(grad_col_d[sphere_idx * cam.n_channels + c_id]),
            weight * grad_im_l[c_id]);
      }
      PULSAR_LOG_DEV_APIX(
          PULSAR_LOG_GRAD,
          "grad|nerf color. grad_idx(%d), sphere_idx(%d), t(%.5f), "
          "sigma(%.5f), att(%.5f), alpha(%.5f), "
          "T(%.5f), weight(%.5f), "
          "result(%.5f, %.5f, %.5f), "
          "grad_im(%.5f, %.5f, %.5f) \n",
          grad_idx, sphere_idx, t, 
          sigma, att, 1.f - att,
          light_intensity, weight,
          result[0], result[1], result[2],
          grad_im_l[0], grad_im_l[1], grad_im_l[2]);   
      light_intensity *= att;
      accum += weight * total_color;
    }
  }
  float total_bg = 0.f;
  for (uint c_id = 0; c_id < cam.n_channels; ++c_id) {
    total_bg += bg_col[c_id] * grad_im_l[c_id];
  }
  accum += light_intensity * total_bg;
  // PASS 2
  light_intensity = 1.f;
  for (int grad_idx = 0; grad_idx < n_track; ++grad_idx) {
    int sphere_idx;
    FASI(forw_info_d[fwi_loc + 3 + 2 * grad_idx], sphere_idx);
    PASSERT(
        sphere_idx == -1 ||
        sphere_idx >= 0 && static_cast<uint>(sphere_idx) < num_balls);
    float t = forw_info_d[fwi_loc + 3 + 2 * grad_idx + 1]; // sphere depth
    if (sphere_idx >= 0 && t >= cam.min_dist) {
      float t_next;
      if (grad_idx < n_track - 1) {
        t_next = 
          forw_info_d[fwi_loc + 3 + 2 * (grad_idx + 1) + 1] == -1.f 
          ? MAX_FLOAT
          : forw_info_d[fwi_loc + 3 + 2 * (grad_idx + 1) + 1];
      } else {
        t_next = MAX_FLOAT;
      }
      if (t_next == MAX_FLOAT)
        // For the last sphere, the opacity gradient is always zero.
        // So we can safely skip it. This is a workaround to avoid 
        // numeric error caused by exp(- delta_t * sigma)
        break;
      float delta_t = FMIN(FABS(t_next - t), 1e10);
      float sigma = opacity == NULL ? MAX_FLOAT : opacity[sphere_idx];
      // Do not use FEXP() here. The error is not marginal!
      float att = exp(- delta_t * sigma);
      float weight = light_intensity * (1.f - att);
      float const* const col_ptr =
        cam.n_channels > 3 ? di_d[sphere_idx].color_union.ptr : &di_d[sphere_idx].first_color;
      float total_color = 0.f;
      for (uint c_id = 0; c_id < cam.n_channels; ++c_id) {
        total_color += col_ptr[c_id] * grad_im_l[c_id];
      }
      light_intensity *= att;
      accum -= weight * total_color;
      float grad_opy = delta_t * (total_color * light_intensity - accum);
      ATOMICADD(&(grad_opy_d[sphere_idx]), grad_opy);
      PULSAR_LOG_DEV_APIX(
          PULSAR_LOG_GRAD,
          "grad|nerf opacity. grad_idx(%d), sphere_idx(%d), t(%.5f), "
          "sigma(%.5f), att(%.5f), alpha(%.5f), "
          "T(%.5f), weight(%.5f), total_color(%.5f), "
          "accum(%.5f), "
          "grad_opy(%.5f), "
          "grad_im(%.5f, %.5f, %.5f) \n",
          grad_idx, sphere_idx, t, 
          sigma, att, 1.f - att,
          light_intensity, weight, total_color,
          accum,
          grad_opy,
          grad_im_l[0], grad_im_l[1], grad_im_l[2]);  
    }
  }
  END_PARALLEL_2D_NORET();
};


template <bool DEV>
GLOBAL void calc_gradients(
    const CamInfo cam, /** Camera in world coordinates. */
    float const* const RESTRICT grad_im, /** The gradient image. */
    const float
        gamma, /** The transparency parameter used in the forward pass. */
    float3 const* const RESTRICT vert_poss, /** Vertex position vector. */
    float const* const RESTRICT vert_cols, /** Vertex color vector. */
    float const* const RESTRICT vert_rads, /** Vertex radius vector. */
    float const* const RESTRICT opacity, /** Vertex opacity. */
    const uint num_balls, /** Number of balls. */
    float const* const RESTRICT result_d, /** Result image. */
    float const* const RESTRICT forw_info_d, /** Forward pass info. */
    DrawInfo const* const RESTRICT di_d, /** Draw information. */
    IntersectInfo const* const RESTRICT ii_d, /** Intersect information. */
    // Mode switches.
    const bool calc_grad_pos,
    const bool calc_grad_col,
    const bool calc_grad_rad,
    const bool calc_grad_cam,
    const bool calc_grad_opy,
    // Out variables.
    float* const RESTRICT grad_rad_d, /** Radius gradients. */
    float* const RESTRICT grad_col_d, /** Color gradients. */
    float3* const RESTRICT grad_pos_d, /** Position gradients. */
    CamGradInfo* const RESTRICT grad_cam_buf_d, /** Camera gradient buffer. */
    float* const RESTRICT grad_opy_d, /** Opacity gradient buffer. */
    int* const RESTRICT
        grad_contributed_d, /** Gradient contribution counter. */
    // Infrastructure.
    const int n_track,
    const uint offs_x,
    const uint offs_y /** Debug offsets. */
) {
  uint limit_x = cam.film_width, limit_y = cam.film_height;
  if (offs_x != 0) {
    // We're in debug mode.
    limit_x = 1;
    limit_y = 1;
  }
  GET_PARALLEL_IDS_2D(coord_x_base, coord_y_base, limit_x, limit_y);
  // coord_x_base and coord_y_base are in the film coordinate system.
  // We now need to translate to the aperture coordinate system. If
  // the principal point was shifted left/up nothing has to be
  // subtracted - only shift needs to be added in case it has been
  // shifted down/right.
  const uint film_coord_x = coord_x_base + offs_x;
  const uint ap_coord_x = film_coord_x +
      2 * static_cast<uint>(std::max(0, cam.principal_point_offset_x));
  const uint film_coord_y = coord_y_base + offs_y;
  const uint ap_coord_y = film_coord_y +
      2 * static_cast<uint>(std::max(0, cam.principal_point_offset_y));
  const float3 ray_dir = /** Ray cast through the pixel, normalized. */
      cam.pixel_0_0_center + ap_coord_x * cam.pixel_dir_x +
      ap_coord_y * cam.pixel_dir_y;
  const float norm_ray_dir = length(ray_dir);
  // ray_dir_norm *must* be calculated here in the same way as in the draw
  // function to have the same values withno other numerical instabilities
  // (for example, ray_dir * FRCP(norm_ray_dir) does not work)!
  float3 ray_dir_norm; /** Ray cast through the pixel, normalized. */
  float2 projected_ray; /** Ray intersection with the sensor. */
  if (cam.orthogonal_projection) {
    ray_dir_norm = cam.sensor_dir_z;
    projected_ray.x = static_cast<float>(ap_coord_x);
    projected_ray.y = static_cast<float>(ap_coord_y);
  } else {
    ray_dir_norm = normalize(
        cam.pixel_0_0_center + ap_coord_x * cam.pixel_dir_x +
        ap_coord_y * cam.pixel_dir_y);
    // This is a reasonable assumption for normal focal lengths and image sizes.
    PASSERT(FABS(ray_dir_norm.z) > FEPS);
    projected_ray.x = ray_dir_norm.x / ray_dir_norm.z * cam.focal_length;
    projected_ray.y = ray_dir_norm.y / ray_dir_norm.z * cam.focal_length;
  }
  float* result = const_cast<float*>(
      result_d + film_coord_y * cam.film_width * cam.n_channels +
      film_coord_x * cam.n_channels);
  const float* grad_im_l = grad_im +
      film_coord_y * cam.film_width * cam.n_channels +
      film_coord_x * cam.n_channels;
  // For writing...
  float3 grad_pos;
  float grad_rad, grad_opy;
  CamGradInfo grad_cam_local = CamGradInfo();
  // Set up shared infrastructure.
  const int fwi_loc = film_coord_y * cam.film_width * (3 + 2 * n_track) +
      film_coord_x * (3 + 2 * n_track);
  float sm_m = forw_info_d[fwi_loc];
  float sm_d = forw_info_d[fwi_loc + 1];
  PULSAR_LOG_DEV_APIX(
      PULSAR_LOG_GRAD,
      "grad|sm_m: %f, sm_d: %f, result: "
      "%f, %f, %f; grad_im: %f, %f, %f.\n",
      sm_m,
      sm_d,
      result[0],
      result[1],
      result[2],
      grad_im_l[0],
      grad_im_l[1],
      grad_im_l[2]);
  // Start processing.
  for (int grad_idx = 0; grad_idx < n_track; ++grad_idx) {
    int sphere_idx;
    FASI(forw_info_d[fwi_loc + 3 + 2 * grad_idx], sphere_idx);
    PASSERT(
        sphere_idx == -1 ||
        sphere_idx >= 0 && static_cast<uint>(sphere_idx) < num_balls);
    if (sphere_idx >= 0) {
      // TODO: make more efficient.
      grad_pos = make_float3(0.f, 0.f, 0.f);
      grad_rad = 0.f;
      grad_cam_local = CamGradInfo();
      const DrawInfo di = di_d[sphere_idx];
      grad_opy = 0.f;
      draw(
          di,
          opacity == NULL ? 1.f : opacity[sphere_idx],
          cam,
          gamma,
          ray_dir_norm,
          projected_ray,
          // Mode switches.
          false, // Hit Only
          false, // draw only
          calc_grad_pos,
          calc_grad_col,
          calc_grad_rad,
          calc_grad_cam,
          calc_grad_opy,
          // Position info.
          ap_coord_x,
          ap_coord_y,
          sphere_idx,
          // Optional in.
          &ii_d[sphere_idx],
          &ray_dir,
          &norm_ray_dir,
          grad_im_l,
          NULL,
          // In/out
          &sm_d,
          &sm_m,
          result,
          // Optional out.
          NULL,
          NULL,
          &grad_pos,
          grad_col_d + sphere_idx * cam.n_channels,
          &grad_rad,
          &grad_cam_local,
          &grad_opy);
      ATOMICADD(&(grad_rad_d[sphere_idx]), grad_rad);
      // Color has been added directly.
      ATOMICADD_F3(&(grad_pos_d[sphere_idx]), grad_pos);
      ATOMICADD_F3(
          &(grad_cam_buf_d[sphere_idx].cam_pos), grad_cam_local.cam_pos);
      if (!cam.orthogonal_projection) {
        ATOMICADD_F3(
            &(grad_cam_buf_d[sphere_idx].pixel_0_0_center),
            grad_cam_local.pixel_0_0_center);
      }
      ATOMICADD_F3(
          &(grad_cam_buf_d[sphere_idx].pixel_dir_x),
          grad_cam_local.pixel_dir_x);
      ATOMICADD_F3(
          &(grad_cam_buf_d[sphere_idx].pixel_dir_y),
          grad_cam_local.pixel_dir_y);
      ATOMICADD(&(grad_opy_d[sphere_idx]), grad_opy);
      ATOMICADD(&(grad_contributed_d[sphere_idx]), 1);
    }
  }
  END_PARALLEL_2D_NORET();
};

} // namespace Renderer
} // namespace pulsar

#endif
