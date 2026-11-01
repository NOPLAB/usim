// SPDX-FileCopyrightText: 2026 Keita Sekiguchi / nop
// SPDX-License-Identifier: Apache-2.0
// Gazebo Classic laser output canvas uses 0.1 units per output pixel.
// Rasterize each ray at its pixel center, not the upper-left boundary.
varying float tex;

void main()
{
  vec4 pixelCenter = gl_Vertex;
  pixelCenter.y -= 0.05;
  pixelCenter.z -= 0.05;
  gl_Position = gl_ModelViewProjectionMatrix * pixelCenter;
  tex = gl_Vertex.x;
  gl_TexCoord[0] = gl_MultiTexCoord0;
}
