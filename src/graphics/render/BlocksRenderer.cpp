#include "BlocksRenderer.hpp"

#include "graphics/core/Mesh.hpp"
#include "graphics/commons/Model.hpp"
#include "maths/UVRegion.hpp"
#include "constants.hpp"
#include "content/Content.hpp"
#include "voxels/Chunks.hpp"
#include "lighting/Lightmap.hpp"
#include "frontend/ContentGfxCache.hpp"

const glm::vec3 BlocksRenderer::SUN_VECTOR(0.528265, 0.833149, -0.163704);
const float DIRECTIONAL_LIGHT_FACTOR = 0.3f;

namespace {
static constexpr float K_CHUNK_CENTER_BIAS = 0.5f;
static constexpr float K_AO_NORMAL_PUSH = 0.75f;

static inline void expand_aabb_point(AABB& aabb, bool& init, const glm::vec3& p) {
    if (!init) { aabb.a = aabb.b = p; init = true; } else { aabb.addPoint(p); }
}

static inline float apply_directional_factor(float d) {
    return (1.0f - DIRECTIONAL_LIGHT_FACTOR) + d * DIRECTIONAL_LIGHT_FACTOR;
}

static inline void expand_aabb_4(
    AABB& aabb, bool& init,
    const glm::vec3& p0, const glm::vec3& p1,
    const glm::vec3& p2, const glm::vec3& p3
) {
    expand_aabb_point(aabb, init, p0);
    expand_aabb_point(aabb, init, p1);
    expand_aabb_point(aabb, init, p2);
    expand_aabb_point(aabb, init, p3);
}

static inline void expand_aabb_4_if_needed(
    AABB& aabb, bool& init, bool densePass,
    const glm::vec3& p0, const glm::vec3& p1,
    const glm::vec3& p2, const glm::vec3& p3
) {
    if (!densePass) {
        expand_aabb_4(aabb, init, p0, p1, p2, p3);
    }
}

static inline void compute_face_points(
    const glm::vec3& coord,
    const glm::vec3& X, const glm::vec3& Y, const glm::vec3& Z,
    float bias,
    glm::vec3& p0, glm::vec3& p1, glm::vec3& p2, glm::vec3& p3
) {
    float s = bias;
    p0 = coord + (-X - Y + Z) * s;
    p1 = coord + ( X - Y + Z) * s;
    p2 = coord + ( X + Y + Z) * s;
    p3 = coord + (-X + Y + Z) * s;
}

static inline void fill_texfaces(
    const ContentGfxCache& cache,
    blockid_t id, uint8_t variantId, bool densePass,
    UVRegion (&out)[6]
) {
    for (int f = 0; f < 6; ++f) out[f] = cache.getRegion(id, variantId, f, densePass);
}
}

BlocksRenderer::BlocksRenderer(
    size_t capacity,
    const Content& content,
    const ContentGfxCache& cache,
    const EngineSettings& settings
) : content(content),
    vertexBuffer(std::make_unique<ChunkVertex[]>(capacity)),
    indexBuffer(std::make_unique<uint32_t[]>(capacity)),
    denseIndexBuffer(std::make_unique<uint32_t[]>(capacity)),
    vertexCount(0),
    vertexOffset(0),
    indexCount(0),
    capacity(capacity),
    cache(cache),
    settings(settings)
{
    voxelsBuffer = std::make_unique<VoxelsVolume>(
        CHUNK_W + voxelBufferPadding*2,
        CHUNK_H,
        CHUNK_D + voxelBufferPadding*2);
    blockDefsCache = content.getIndices()->blocks.getDefs();
}

BlocksRenderer::~BlocksRenderer() = default;

/// Basic vertex add method
void BlocksRenderer::vertex(
    const glm::vec3& coord,
    float u,
    float v,
    const glm::vec4& light,
    const glm::vec3& normal,
    float emission
) {
    vertexBuffer[vertexCount].position = coord;

    vertexBuffer[vertexCount].uv = {u,v};

    vertexBuffer[vertexCount].normal[0] = static_cast<uint8_t>(normal.r * 127 + 128);
    vertexBuffer[vertexCount].normal[1] = static_cast<uint8_t>(normal.g * 127 + 128);
    vertexBuffer[vertexCount].normal[2] = static_cast<uint8_t>(normal.b * 127 + 128);
    vertexBuffer[vertexCount].normal[3] = static_cast<uint8_t>(emission * 255);

    vertexBuffer[vertexCount].color[0] = static_cast<uint8_t>(light.r * 255);
    vertexBuffer[vertexCount].color[1] = static_cast<uint8_t>(light.g * 255);
    vertexBuffer[vertexCount].color[2] = static_cast<uint8_t>(light.b * 255);
    vertexBuffer[vertexCount].color[3] = static_cast<uint8_t>(light.a * 255);

    vertexCount++;
}

void BlocksRenderer::index(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e, uint32_t f) {
    indexBuffer[indexCount++] = static_cast<uint32_t>(vertexOffset + a);
    indexBuffer[indexCount++] = static_cast<uint32_t>(vertexOffset + b);
    indexBuffer[indexCount++] = static_cast<uint32_t>(vertexOffset + c);
    indexBuffer[indexCount++] = static_cast<uint32_t>(vertexOffset + d);
    indexBuffer[indexCount++] = static_cast<uint32_t>(vertexOffset + e);
    indexBuffer[indexCount++] = static_cast<uint32_t>(vertexOffset + f);
    vertexOffset += 4;
}

/// @brief Add face with precalculated lights
void BlocksRenderer::face(
    const glm::vec3& coord,
    float w, float h, float d,
    const glm::vec3& axisX,
    const glm::vec3& axisY,
    const glm::vec3& axisZ,
    const UVRegion& region,
    const glm::vec4(&lights)[4],
    const glm::vec4& tint
) {
    if (vertexCount + 4 >= capacity) {
        overflow = true;
        return;
    }
    auto X = axisX * w;
    auto Y = axisY * h;
    auto Z = axisZ * d;
    glm::vec3 p0, p1, p2, p3;
    compute_face_points(coord, X, Y, Z, K_CHUNK_CENTER_BIAS, p0, p1, p2, p3);
    vertex(p0, region.u1, region.v1, lights[0] * tint, axisZ, 0);
    vertex(p1, region.u2, region.v1, lights[1] * tint, axisZ, 0);
    vertex(p2, region.u2, region.v2, lights[2] * tint, axisZ, 0);
    vertex(p3, region.u1, region.v2, lights[3] * tint, axisZ, 0);
    index(0, 1, 3, 1, 2, 3);

    // Expand local opaque AABB while vertices are still in chunk-local space
    expand_aabb_4_if_needed(localAabb, localAabbInit, densePass, p0, p1, p2, p3);
}

void BlocksRenderer::vertexAO(
    const glm::vec3& coord,
    float u, float v,
    const glm::vec4& tint,
    float normalHalfLen,
    const glm::vec3& axisX,
    const glm::vec3& axisY,
    const glm::vec3& axisZ
) {
    // Sample AO in world-voxel grid with a slightly longer reach along the normal
    // to avoid sampling the same chunk voxel when faces reside over neighbor chunk.
    auto pos = coord + axisZ * normalHalfLen + (axisX + axisY) * 0.5f;
    auto light = pickSoftLight(
        glm::ivec3(std::round(pos.x), std::round(pos.y), std::round(pos.z)),
        axisX,
        axisY
    );
    vertex(coord, u, v, light * tint, axisZ, 0.0f);
}

void BlocksRenderer::faceAO(
    const glm::vec3& coord,
    const glm::vec3& X,
    const glm::vec3& Y,
    const glm::vec3& Z,
    const UVRegion& region,
    bool lights
) {
    if (vertexCount + 4 >= capacity) {
        overflow = true;
        return;
    }

    if (lights) {
        const auto nZ = glm::normalize(Z);
        float d = apply_directional_factor(glm::dot(nZ, SUN_VECTOR));

        auto axisX = glm::normalize(X);
        auto axisY = glm::normalize(Y);
        auto axisZ = nZ;

        glm::vec4 tint(d);
        const float nh = K_AO_NORMAL_PUSH; // push AO sample a bit farther along normal
        glm::vec3 p0, p1, p2, p3;
        compute_face_points(coord, X, Y, Z, K_CHUNK_CENTER_BIAS, p0, p1, p2, p3);
        vertexAO(p0, region.u1, region.v1, tint, nh, axisX, axisY, axisZ);
        vertexAO(p1, region.u2, region.v1, tint, nh, axisX, axisY, axisZ);
        vertexAO(p2, region.u2, region.v2, tint, nh, axisX, axisY, axisZ);
        vertexAO(p3, region.u1, region.v2, tint, nh, axisX, axisY, axisZ);
        expand_aabb_4_if_needed(localAabb, localAabbInit, densePass, p0, p1, p2, p3);
    } else {
        auto axisZ = glm::normalize(Z);
        glm::vec4 tint(1.0f);
        glm::vec3 p0, p1, p2, p3;
        compute_face_points(coord, X, Y, Z, K_CHUNK_CENTER_BIAS, p0, p1, p2, p3);
        vertex(p0, region.u1, region.v1, tint, axisZ, 1);
        vertex(p1, region.u2, region.v1, tint, axisZ, 1);
        vertex(p2, region.u2, region.v2, tint, axisZ, 1);
        vertex(p3, region.u1, region.v2, tint, axisZ, 1);
        expand_aabb_4_if_needed(localAabb, localAabbInit, densePass, p0, p1, p2, p3);
    }
    index(0, 1, 2, 0, 2, 3);
}

void BlocksRenderer::face(
    const glm::vec3& coord,
    const glm::vec3& X,
    const glm::vec3& Y,
    const glm::vec3& Z,
    const UVRegion& region,
    glm::vec4 tint,
    bool lights
) {
    if (vertexCount + 4 >= capacity) {
        overflow = true;
        return;
    }

    const auto nZ = glm::normalize(Z);
    if (lights) {
        float d = apply_directional_factor(glm::dot(nZ, SUN_VECTOR));
        tint *= d;
    }
    const auto nZ2 = lights ? nZ : Z;
    glm::vec3 p0, p1, p2, p3;
    compute_face_points(coord, X, Y, Z, K_CHUNK_CENTER_BIAS, p0, p1, p2, p3);
    vertex(p0, region.u1, region.v1, tint, nZ2, lights ? 0 : 1);
    vertex(p1, region.u2, region.v1, tint, nZ2, lights ? 0 : 1);
    vertex(p2, region.u2, region.v2, tint, nZ2, lights ? 0 : 1);
    vertex(p3, region.u1, region.v2, tint, nZ2, lights ? 0 : 1);
    expand_aabb_4_if_needed(localAabb, localAabbInit, densePass, p0, p1, p2, p3);
    index(0, 1, 2, 0, 2, 3);
}

void BlocksRenderer::blockXSprite(
    int x, int y, int z,
    const glm::vec3& size,
    const UVRegion& texface1,
    const UVRegion& texface2,
    float spread
) {
    glm::vec4 lights1[] {
        pickSoftLight({x, y + 1, z}, {1, 0, 0}, {0, 1, 0}),
        pickSoftLight({x + 1, y + 1, z}, {1, 0, 0}, {0, 1, 0}),
        pickSoftLight({x + 1, y + 1, z}, {1, 0, 0}, {0, 1, 0}),
        pickSoftLight({x, y + 1, z}, {1, 0, 0}, {0, 1, 0})
    };
    glm::vec4 lights2[] {
        pickSoftLight({x, y + 1, z}, {-1, 0, 0}, {0, 1, 0}),
        pickSoftLight({x - 1, y + 1, z}, {-1, 0, 0}, {0, 1, 0}),
        pickSoftLight({x - 1, y + 1, z}, {-1, 0, 0}, {0, 1, 0}),
        pickSoftLight({x, y + 1, z}, {-1, 0, 0}, {0, 1, 0})
    };
    randomizer.setSeed((x * 52321) ^ (z * 389) ^ y);
    short rand = randomizer.rand32();

    float xs = ((float)(char)rand / 512) * spread;
    float zs = ((float)(char)(rand >> 8) / 512) * spread;

    const float w = size.x / 1.41f;
    const glm::vec4 tint (0.8f);

    glm::vec3 n(0.0f, 1.0f, 0.0f);

    face({x + xs, y, z + zs}, w, size.y, 0, {-1, 0, 1}, {0, 1, 0}, n,
        texface1, lights2, tint);
    face({x + xs, y, z + zs}, w, size.y, 0, {1, 0, 1}, {0, 1, 0}, n,
        texface1, lights1, tint);

    face({x + xs, y, z + zs}, w, size.y, 0, {-1, 0, -1}, {0, 1, 0}, n,
        texface2, lights2, tint);
    face({x + xs, y, z + zs}, w, size.y, 0, {1, 0, -1}, {0, 1, 0}, n,
        texface2, lights1, tint);
}

// HINT: texture faces order: {east, west, bottom, top, south, north}

/// @brief AABB blocks render method
void BlocksRenderer::blockAABB(
    const glm::ivec3& icoord,
    const UVRegion(&texfaces)[6],
    const Block* block,
    ubyte rotation,
    bool lights,
    bool ao
) {
    if (block->hitboxes.empty()) {
        return;
    }
    AABB hitbox = block->hitboxes[0];
    for (const auto& box : block->hitboxes) {
        hitbox.a = glm::min(hitbox.a, box.a);
        hitbox.b = glm::max(hitbox.b, box.b);
    }
    auto size = hitbox.size();
    glm::vec3 X(1, 0, 0);
    glm::vec3 Y(0, 1, 0);
    glm::vec3 Z(0, 0, 1);
    glm::vec3 coord(icoord);
    if (block->rotatable) {
        auto& rotations = block->rotations;
        auto& orient = rotations.variants[rotation];
        X = orient.axes[0];
        Y = orient.axes[1];
        Z = orient.axes[2];
        orient.transform(hitbox);
    }
    coord -= glm::vec3(0.5f) - hitbox.center();

    if (ao) {
        faceAO(coord,  X*size.x,  Y*size.y,  Z*size.z, texfaces[5], lights); // north
        faceAO(coord, -X*size.x,  Y*size.y, -Z*size.z, texfaces[4], lights); // south

        faceAO(coord,  X*size.x, -Z*size.z,  Y*size.y, texfaces[3], lights); // top
        faceAO(coord, -X*size.x, -Z*size.z, -Y*size.y, texfaces[2], lights); // bottom

        faceAO(coord, -Z*size.z,  Y*size.y,  X*size.x, texfaces[1], lights); // west
        faceAO(coord,  Z*size.z,  Y*size.y, -X*size.x, texfaces[0], lights); // east
    } else {
        auto tint = pickLight(icoord);
        face(coord,  X*size.x,  Y*size.y,  Z*size.z, texfaces[5], tint, lights); // north
        face(coord, -X*size.x,  Y*size.y, -Z*size.z, texfaces[4], tint, lights); // south

        face(coord,  X*size.x, -Z*size.z,  Y*size.y, texfaces[3], tint, lights); // top
        face(coord, -X*size.x, -Z*size.z, -Y*size.y, texfaces[2], tint, lights); // bottom

        face(coord, -Z*size.z,  Y*size.y,  X*size.x, texfaces[1], tint, lights); // west
        face(coord,  Z*size.z,  Y*size.y, -X*size.x, texfaces[0], tint, lights); // east
    }
}

static bool is_aligned(const glm::vec3& v, float e = 1e-6f) {
    if (std::abs(v.y) < e && std::abs(v.z) < e && std::abs(v.x) > e) {
        return true;
    }
    if (std::abs(v.x) < e && std::abs(v.z) < e && std::abs(v.y) > e) {
        return true;
    }
    if (std::abs(v.x) < e && std::abs(v.y) < e && std::abs(v.z) > e) {
        return true;
    }
    return false;
}

void BlocksRenderer::blockCustomModel(
    const glm::ivec3& icoord, const Block& block, blockstate states, bool lights, bool ao
) {
    const auto& variant = block.getVariantByBits(states.userbits);
    glm::vec3 X(1, 0, 0);
    glm::vec3 Y(0, 1, 0);
    glm::vec3 Z(0, 0, 1);
    glm::vec3 coord(icoord);
    if (block.rotatable) {
        auto& rotations = block.rotations;
        CoordSystem orient = rotations.variants[states.rotation];
        X = orient.axes[0];
        Y = orient.axes[1];
        Z = orient.axes[2];
    }

    const auto& model = cache.getModel(block.rt.id, block.getVariantIndex(states.userbits));
    for (const auto& mesh : model.meshes) {
        if (vertexCount + mesh.vertices.size() >= capacity) {
            overflow = true;
            return;
        }
        for (int triangle = 0; triangle < mesh.vertices.size() / 3; triangle++) {
            auto r = mesh.vertices[triangle * 3 + (triangle % 2) * 2].coord -
                     mesh.vertices[triangle * 3 + 1].coord;
            r = r.x * X + r.y * Y + r.z * Z;
            r = glm::normalize(r);

            const auto& v0 = mesh.vertices[triangle * 3];
            auto n = v0.normal.x * X + v0.normal.y * Y + v0.normal.z * Z;

            auto vp = (mesh.vertices[triangle * 3].coord +
                       mesh.vertices[triangle * 3 + 1].coord +
                       mesh.vertices[triangle * 3 + 2].coord) *
                          0.3333f -
                      0.5f;
            vp = vp.x * X + vp.y * Y + vp.z * Z;

            if (!isOpen(glm::floor(coord + vp + 0.5f + n * 1e-3f), block, variant) && is_aligned(n)) {
                continue;
            }

            float d = apply_directional_factor(glm::dot(n, SUN_VECTOR));
            glm::vec3 t = glm::cross(r, n);

            for (int i = 0; i < 3; i++) {
                const auto& vertex = mesh.vertices[triangle * 3 + i];
                const auto& vcoord = vertex.coord - 0.5f;

                glm::vec4 aoColor {1.0f, 1.0f, 1.0f, 1.0f};
                if (mesh.shading && ao) {
                    auto p = coord + vcoord.x * X + vcoord.y * Y +
                             vcoord.z * Z + r * 0.5f + t * 0.5f + n * 0.5f;
                    aoColor = pickSoftLight(p.x, p.y, p.z, glm::ivec3(r), glm::ivec3(t));
                }
                auto pLocal = coord + vcoord.x * X + vcoord.y * Y + vcoord.z * Z;
                this->vertex(
                    pLocal,
                    vertex.uv.x,
                    vertex.uv.y,
                    mesh.shading ? (glm::vec4(d, d, d, d) * aoColor) : glm::vec4(1, 1, 1, d),
                    n,
                    mesh.shading ? 0.0f : 1.0
                );
                if (!densePass) {
                    expand_aabb_point(localAabb, localAabbInit, pLocal);
                }
                indexBuffer[indexCount++] = vertexOffset++;
            }
        }
    }
}

/* Fastest solid shaded blocks render method */
void BlocksRenderer::blockCube(
    const glm::ivec3& coord,
    const UVRegion(&texfaces)[6],
    const Block& block,
    blockstate states,
    bool lights,
    bool ao
) {
    const auto& variant = block.getVariantByBits(states.userbits);
    glm::ivec3 X(1, 0, 0);
    glm::ivec3 Y(0, 1, 0);
    glm::ivec3 Z(0, 0, 1);

    if (block.rotatable) {
        auto& rotations = block.rotations;
        auto& orient = rotations.variants[states.rotation];
        X = orient.axes[0];
        Y = orient.axes[1];
        Z = orient.axes[2];
    }

    if (ao) {
        if (isOpen(coord + Z, block, variant)) {
            faceAO(coord, X, Y, Z, texfaces[5], lights);
        }
        if (isOpen(coord - Z, block, variant)) {
            faceAO(coord, -X, Y, -Z, texfaces[4], lights);
        }
        if (isOpen(coord + Y, block, variant)) {
            faceAO(coord, X, -Z, Y, texfaces[3], lights);
        }
        if (isOpen(coord - Y, block, variant)) {
            faceAO(coord, X, Z, -Y, texfaces[2], lights);
        }
        if (isOpen(coord + X, block, variant)) {
            faceAO(coord, -Z, Y, X, texfaces[1], lights);
        }
        if (isOpen(coord - X, block, variant)) {
            faceAO(coord, Z, Y, -X, texfaces[0], lights);
        }
    } else {
        if (isOpen(coord + Z, block, variant)) {
            face(coord, X, Y, Z, texfaces[5], pickLight(coord + Z), lights);
        }
        if (isOpen(coord - Z, block, variant)) {
            face(coord, -X, Y, -Z, texfaces[4], pickLight(coord - Z), lights);
        }
        if (isOpen(coord + Y, block, variant)) {
            face(coord, X, -Z, Y, texfaces[3], pickLight(coord + Y), lights);
        }
        if (isOpen(coord - Y, block, variant)) {
            face(coord, X, Z, -Y, texfaces[2], pickLight(coord - Y), lights);
        }
        if (isOpen(coord + X, block, variant)) {
            face(coord, -Z, Y, X, texfaces[1], pickLight(coord + X), lights);
        }
        if (isOpen(coord - X, block, variant)) {
            face(coord, Z, Y, -X, texfaces[0], pickLight(coord - X), lights);
        }
    }
}

bool BlocksRenderer::isOpenForLight(int x, int y, int z) const {
    blockid_t id = voxelsBuffer->pickBlockId(chunk->x * CHUNK_W + x,
                                             y,
                                             chunk->z * CHUNK_D + z);
    if (id == BLOCK_VOID) {
        return false;
    }
    const Block& block = *blockDefsCache[id];
    if (block.lightPassing) {
        return true;
    }
    return !id;
}

glm::vec4 BlocksRenderer::pickLight(int x, int y, int z) const {
    if (isOpenForLight(x, y, z)) {
        light_t light = voxelsBuffer->pickLight(chunk->x * CHUNK_W + x, y,
                                                chunk->z * CHUNK_D + z);
        return glm::vec4(Lightmap::extract(light, 0),
                         Lightmap::extract(light, 1),
                         Lightmap::extract(light, 2),
                         Lightmap::extract(light, 3)) / 15.0f;
    } else {
        return glm::vec4(0.0f);
    }
}

glm::vec4 BlocksRenderer::pickLight(const glm::ivec3& coord) const {
    return pickLight(coord.x, coord.y, coord.z);
}

glm::vec4 BlocksRenderer::pickSoftLight(
    const glm::ivec3& coord, const glm::ivec3& right, const glm::ivec3& up
) const {
    return (pickLight(coord) +
            pickLight(coord - right) +
            pickLight(coord - right - up) +
            pickLight(coord - up)) * 0.25f;
}

glm::vec4 BlocksRenderer::pickSoftLight(
    float x, float y, float z, const glm::ivec3& right, const glm::ivec3& up
) const {
    return pickSoftLight({
        static_cast<int>(std::round(x)),
        static_cast<int>(std::round(y)),
        static_cast<int>(std::round(z))},
        right, up);
}

void BlocksRenderer::render(
    const voxel* voxels, const int beginEnds[256][2]
) {
    bool denseRender = this->denseRender;
    bool densePass = this->densePass;
    bool enableAO = settings.graphics.softLighting.get();
    for (const auto drawGroup : *content.drawGroups) {
        int begin = beginEnds[drawGroup][0];
        if (begin == 0) {
            continue;
        }
        int end = beginEnds[drawGroup][1];
        for (int i = begin-1; i <= end; i++) {
            const voxel& vox = voxels[i];
            blockid_t id = vox.id;
            blockstate state = vox.state;
            const auto& def = *blockDefsCache[id];
            uint8_t variantId = def.getVariantIndex(state.userbits);
            const auto& variant = def.getVariant(variantId);
            if (id == 0 || variant.drawGroup != drawGroup || state.segment) {
                continue;
            }
            if (denseRender != (variant.culling == CullingMode::OPTIONAL)) {
                continue;
            }
            if (def.translucent) {
                continue;
            }
            UVRegion texfaces[6];
            fill_texfaces(cache, id, variantId, densePass, texfaces);
            int x = i % CHUNK_W;
            int y = i / (CHUNK_D * CHUNK_W);
            int z = (i / CHUNK_D) % CHUNK_W;
            switch (def.getModel(state.userbits).type) {
                case BlockModelType::BLOCK:
                    blockCube({x, y, z}, texfaces, def, vox.state, !def.shadeless,
                              def.ambientOcclusion && enableAO);
                    break;
                case BlockModelType::XSPRITE: {
                    if (!denseRender)
                    blockXSprite(x, y, z, glm::vec3(1.0f),
                                texfaces[FACE_MX], texfaces[FACE_MZ], 1.0f);
                    break;
                }
                case BlockModelType::AABB: {
                    if (!denseRender)
                    blockAABB({x, y, z}, texfaces, &def, vox.state.rotation,
                              !def.shadeless, def.ambientOcclusion && enableAO);
                    break;
                }
                case BlockModelType::CUSTOM: {
                    if (!denseRender)
                    blockCustomModel(
                        {x, y, z},
                        def,
                        vox.state,
                        !def.shadeless,
                        def.ambientOcclusion && enableAO
                    );
                    break;
                }
                default:
                    break;
            }
            if (overflow) {
                return;
            }
        }
    }
}

SortingMeshData BlocksRenderer::renderTranslucent(
    const voxel* voxels, int beginEnds[256][2]
) {
    SortingMeshData sortingMesh {{}};

    AABB aabb {};
    bool aabbInit = false;
    size_t totalSize = 0;

    bool densePass = this->densePass;
    bool enableAO = settings.graphics.softLighting.get();
    for (const auto drawGroup : *content.drawGroups) {
        int begin = beginEnds[drawGroup][0];
        if (begin == 0) {
            continue;
        }
        int end = beginEnds[drawGroup][1];
        for (int i = begin-1; i <= end; i++) {
            const voxel& vox = voxels[i];
            blockid_t id = vox.id;
            blockstate state = vox.state;
            const auto& def = *blockDefsCache[id];
            uint8_t variantId = def.getVariantIndex(state.userbits);
            const auto& variant = def.getVariant(variantId);
            if (id == 0 || variant.drawGroup != drawGroup || state.segment) {
                continue;
            }
            if (!def.translucent) {
                continue;
            }
            UVRegion texfaces[6];
            fill_texfaces(cache, id, variantId, densePass, texfaces);
            int x = i % CHUNK_W;
            int y = i / (CHUNK_D * CHUNK_W);
            int z = (i / CHUNK_D) % CHUNK_W;
            switch (def.getModel(state.userbits).type) {
                case BlockModelType::BLOCK:
                    blockCube({x, y, z}, texfaces, def, vox.state, !def.shadeless,
                              def.ambientOcclusion && enableAO);
                    break;
                case BlockModelType::XSPRITE: {
                    blockXSprite(x, y, z, glm::vec3(1.0f),
                                texfaces[FACE_MX], texfaces[FACE_MZ], 1.0f);
                    break;
                }
                case BlockModelType::AABB: {
                    blockAABB(
                        {x, y, z},
                        texfaces,
                        &def,
                        vox.state.rotation,
                        !def.shadeless,
                        def.ambientOcclusion && enableAO
                    );
                    break;
                }
                case BlockModelType::CUSTOM: {
                    blockCustomModel(
                        {x, y, z},
                        def,
                        vox.state,
                        !def.shadeless,
                        def.ambientOcclusion && enableAO
                    );
                    break;
                }
                default:
                    break;
            }
            if (vertexCount == 0) {
                continue;
            }
            SortingMeshEntry entry {
                glm::vec3(
                    x + chunk->x * CHUNK_W + 0.5f,
                    y + 0.5f,
                    z + chunk->z * CHUNK_D + 0.5f
                ),
                util::Buffer<ChunkVertex>(indexCount), 0};

            totalSize += entry.vertexData.size();

            for (int j = 0; j < indexCount; j++) {
                std::memcpy(
                    entry.vertexData.data() + j,
                    vertexBuffer.get() + indexBuffer[j],
                    sizeof(ChunkVertex)
                );
                ChunkVertex& vertex = entry.vertexData[j];

                if (!aabbInit) {
                    aabbInit = true;
                    aabb.a = aabb.b = vertex.position;
                } else {
                    aabb.addPoint(vertex.position);
                }

                // also widen overall local AABB for translucent geometry
                expand_aabb_point(localAabb, localAabbInit, vertex.position);

                vertex.position.x += chunk->x * CHUNK_W + 0.5f;
                vertex.position.y += 0.5f;
                vertex.position.z += chunk->z * CHUNK_D + 0.5f;
            }
            sortingMesh.entries.push_back(std::move(entry));
            vertexCount = 0;
            vertexOffset = indexCount = 0;
        }
    }

    // additional powerful optimization
    auto size = aabb.size();
    if ((size.y < 0.01f || size.x < 0.01f || size.z < 0.01f) &&
         sortingMesh.entries.size() > 1) {
        SortingMeshEntry newEntry {
            sortingMesh.entries[0].position,
            util::Buffer<ChunkVertex>(totalSize),
            0
        };
        size_t offset = 0;
        for (const auto& entry : sortingMesh.entries) {
            std::memcpy(
                newEntry.vertexData.data() + offset,
                entry.vertexData.data(),
                entry.vertexData.size() * sizeof(ChunkVertex)
            );
            offset += entry.vertexData.size();
        }
        return SortingMeshData {{std::move(newEntry)}};
    }
    return sortingMesh;
}

void BlocksRenderer::build(const Chunk* chunk, const Chunks* chunks) {
    this->chunk = chunk;
    // reset local AABB accumulation
    localAabbInit = false;
    localAabb = AABB{glm::vec3(0.0f), glm::vec3(0.0f)};
    voxelsBuffer->setPosition(
        chunk->x * CHUNK_W - voxelBufferPadding, 0,
        chunk->z * CHUNK_D - voxelBufferPadding);
    chunks->getVoxels(*voxelsBuffer, settings.graphics.backlight.get());

    if (voxelsBuffer->pickBlockId(
        chunk->x * CHUNK_W, 0, chunk->z * CHUNK_D
    ) == BLOCK_VOID) {
        cancelled = true;
        return;
    }
    const voxel* voxels = chunk->voxels;

    int totalBegin = chunk->bottom * (CHUNK_W * CHUNK_D);
    int totalEnd = chunk->top * (CHUNK_W * CHUNK_D);

    int beginEnds[256][2] {};
    for (int i = totalBegin; i < totalEnd; i++) {
        const voxel& vox = voxels[i];
        blockid_t id = vox.id;
        const auto& def = *blockDefsCache[id];
        const auto& variant = def.getVariantByBits(vox.state.userbits);

        if (beginEnds[variant.drawGroup][0] == 0) {
            beginEnds[variant.drawGroup][0] = i+1;
        }
        beginEnds[variant.drawGroup][1] = i;
    }
    cancelled = false;

    overflow = false;
    vertexCount = 0;
    vertexOffset = indexCount = 0;

    denseRender = false;
    densePass = false;

    sortingMesh = renderTranslucent(voxels, beginEnds);

    overflow = false;
    vertexCount = 0;
    vertexOffset = 0;
    indexCount = 0;
    denseIndexCount = 0;

    denseRender = false; //settings.graphics.denseRender.get();
    densePass = false;
    render(voxels, beginEnds);

    size_t endIndex = indexCount;
    
    denseRender = true;
    densePass = true;
    render(voxels, beginEnds);

    denseIndexCount = indexCount;
    for (size_t i = 0; i < denseIndexCount; i++) {
        denseIndexBuffer[i] = indexBuffer[i];
    }

    indexCount = endIndex;
    densePass = false;
    render(voxels, beginEnds);
}

ChunkMeshData BlocksRenderer::createMesh() {
    return ChunkMeshData {
        MeshData(
            util::Buffer(vertexBuffer.get(), vertexCount),
            std::vector<util::Buffer<uint32_t>> {
                util::Buffer(indexBuffer.get(), indexCount),
                util::Buffer(denseIndexBuffer.get(), denseIndexCount),
            },
            util::Buffer(
                ChunkVertex::ATTRIBUTES, sizeof(ChunkVertex::ATTRIBUTES) / sizeof(VertexAttribute)
            )
        ),
        std::move(sortingMesh),
        localAabbInit ? localAabb : AABB{glm::vec3(0.0f), glm::vec3(0.0f)}
    };
}

ChunkMesh BlocksRenderer::render(const Chunk *chunk, const Chunks *chunks) {
    build(chunk, chunks);

    return ChunkMesh{std::make_unique<Mesh<ChunkVertex>>(
        vertexBuffer.get(), vertexCount, 
        std::vector<IndexBufferData> {
            IndexBufferData {indexBuffer.get(), indexCount},
            IndexBufferData {denseIndexBuffer.get(), denseIndexCount},
        }
    ), std::move(sortingMesh)};
}

VoxelsVolume* BlocksRenderer::getVoxelsBuffer() const {
    return voxelsBuffer.get();
}

size_t BlocksRenderer::getMemoryConsumption() const {
    size_t volume = voxelsBuffer->getW() * voxelsBuffer->getH() * voxelsBuffer->getD();
    return capacity * (sizeof(ChunkVertex) + sizeof(uint32_t) * 2) + volume * (sizeof(voxel) + sizeof(light_t));
}
