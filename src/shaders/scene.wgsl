// Hello-VR scene: a rotating cube above a floor quad, with "Hello, World!"
// floating above it. All geometry is generated from the vertex index, so no
// vertex buffers are needed, and the text is drawn from a bitmap font baked
// into the fragment shader, so no textures or descriptor sets are needed.
// Multiview renders both eyes in one pass; `view_index` selects the eye.

struct Push {
    // Per-eye (projection * view * model) matrices.
    mvp: array<mat4x4<f32>, 2>,
}

var<immediate> pc: Push;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
    // 0 = solid colour, 1 = text.
    @location(2) @interpolate(flat) kind: u32,
}

// Two CCW triangles covering [-1, 1]^2.
const QUAD = array<vec2<f32>, 6>(
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
);

// Per-face normal, tangent and bitangent with tangent x bitangent == normal,
// so every face winds counter-clockwise when seen from outside.
const NORMAL = array<vec3<f32>, 6>(
    vec3(1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0), vec3(0.0, -1.0, 0.0),
    vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, -1.0),
);
const TANGENT = array<vec3<f32>, 6>(
    vec3(0.0, 0.0, -1.0), vec3(0.0, 0.0, 1.0),
    vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0),
    vec3(1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0),
);
const BITANGENT = array<vec3<f32>, 6>(
    vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, -1.0), vec3(0.0, 0.0, 1.0),
    vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0),
);

const CUBE_VERTS: u32 = 36u;
const FLOOR_VERTS: u32 = 6u;
const TEXT_START: u32 = CUBE_VERTS + FLOOR_VERTS;

// Text layout: 5x7 pixel glyphs in 6x7 cells (one column of spacing).
const TEXT_LEN: u32 = 13u;
const GLYPH_W: u32 = 5u;
const GLYPH_H: u32 = 7u;
const CELL_W: u32 = 6u;
// Half extents of the text quad in model units, matching the cell aspect ratio.
const TEXT_HALF_W: f32 = 0.75;
const TEXT_HALF_H: f32 = TEXT_HALF_W * f32(GLYPH_H) / f32(TEXT_LEN * CELL_W);

// "Hello, World!" as indices into FONT.
const TEXT = array<u32, 13>(0u, 1u, 2u, 2u, 3u, 4u, 5u, 6u, 3u, 7u, 2u, 8u, 9u);

// One row per entry, 5 bits wide, most significant bit on the left.
// Glyph order: H e l o , space W r d !
const FONT = array<u32, 70>(
    // 'H'
    0x11u, 0x11u, 0x11u, 0x1fu, 0x11u, 0x11u, 0x11u,
    // 'e'
    0x00u, 0x00u, 0x0eu, 0x11u, 0x1fu, 0x10u, 0x0eu,
    // 'l'
    0x0cu, 0x04u, 0x04u, 0x04u, 0x04u, 0x04u, 0x0eu,
    // 'o'
    0x00u, 0x00u, 0x0eu, 0x11u, 0x11u, 0x11u, 0x0eu,
    // ','
    0x00u, 0x00u, 0x00u, 0x00u, 0x06u, 0x04u, 0x08u,
    // ' '
    0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u,
    // 'W'
    0x11u, 0x11u, 0x11u, 0x15u, 0x15u, 0x15u, 0x0au,
    // 'r'
    0x00u, 0x00u, 0x16u, 0x19u, 0x10u, 0x10u, 0x10u,
    // 'd'
    0x01u, 0x01u, 0x0du, 0x13u, 0x11u, 0x13u, 0x0du,
    // '!'
    0x04u, 0x04u, 0x04u, 0x04u, 0x04u, 0x00u, 0x04u,
);

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @builtin(view_index) view: u32,
) -> VsOut {
    var quad = QUAD;
    var normal = NORMAL;
    var tangent = TANGENT;
    var bitangent = BITANGENT;

    var out: VsOut;
    var position: vec3<f32>;
    out.kind = 0u;
    out.uv = vec2(0.0);

    if vi < CUBE_VERTS {
        let face = vi / 6u;
        let q = quad[vi % 6u];
        let n = normal[face];
        position = n + q.x * tangent[face] + q.y * bitangent[face];
        out.color = n * 0.5 + vec3(0.5);
    } else if vi < TEXT_START {
        // Floor: 4 m x 4 m quad on y = 0, facing +Y.
        let q = quad[(vi - CUBE_VERTS) % 6u];
        position = vec3(q.x * 2.0, 0.0, -q.y * 2.0);
        out.color = vec3(0.25, 0.25, 0.3);
    } else {
        // Text: quad in the XY plane facing +Z, toward the user.
        let q = quad[(vi - TEXT_START) % 6u];
        position = vec3(q.x * TEXT_HALF_W, q.y * TEXT_HALF_H, 0.0);
        out.uv = q * 0.5 + vec2(0.5);
        out.color = vec3(1.0, 0.9, 0.3);
        out.kind = 1u;
    }

    out.pos = pc.mvp[view] * vec4(position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if in.kind == 1u {
        var text = TEXT;
        var font = FONT;

        // Which cell, and which pixel inside it. uv.y runs bottom to top,
        // font rows run top to bottom.
        let x = in.uv.x * f32(TEXT_LEN);
        let cell = min(u32(x), TEXT_LEN - 1u);
        let col = u32(fract(x) * f32(CELL_W));
        let row = min(u32((1.0 - in.uv.y) * f32(GLYPH_H)), GLYPH_H - 1u);
        if col >= GLYPH_W {
            discard;
        }
        let bits = font[text[cell] * GLYPH_H + row];
        if ((bits >> (GLYPH_W - 1u - col)) & 1u) == 0u {
            discard;
        }
    }
    return vec4(in.color, 1.0);
}
