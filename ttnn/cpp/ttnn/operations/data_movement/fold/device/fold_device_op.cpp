// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fold_device_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::data_movement {

using tt::tt_metal::ShardSpec;

namespace {

// Synthesise an output shard spec when the user requested sharded output without a spec; mirrors
// generate_repeat_shard_spec / _transpose_shard_spec. W/B-shards floor at TILE_WIDTH for NOC alignment.
std::optional<ShardSpec> generate_fold_shard_spec(
    const Tensor& input_tensor, const ttnn::Shape& folded_shape, tt::tt_metal::TensorMemoryLayout requested_layout) {
    const auto grid = input_tensor.device()->compute_with_storage_grid_size();
    const uint32_t total_pixels = folded_shape[0] * folded_shape[1] * folded_shape[2];
    const uint32_t channels = folded_shape[3];
    const auto orientation = input_tensor.memory_config().shard_spec().has_value()
                                 ? input_tensor.memory_config().shard_spec().value().orientation
                                 : tt::tt_metal::ShardOrientation::ROW_MAJOR;

    switch (requested_layout) {
        case tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED: {
            const uint32_t max_cores = grid.x * grid.y;
            uint32_t ncores = std::min(total_pixels, max_cores);
            while (ncores > 0 && total_pixels % ncores != 0) {
                --ncores;
            }
            if (ncores == 0) {
                return std::nullopt;
            }
            const auto core_set = tt::tt_metal::num_cores_to_corerangeset(ncores, grid, true);
            return ShardSpec(core_set, {total_pixels / ncores, channels}, orientation);
        }
        case tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED: {
            // noc_async_write_sharded splits per-pixel along W; conservatively require shard_w ≥ TILE_WIDTH.
            const uint32_t max_cores = grid.x * grid.y;
            const uint32_t max_by_alignment = std::max(1u, channels / tt::constants::TILE_WIDTH);
            uint32_t ncores = std::min({channels, max_cores, max_by_alignment});
            while (ncores > 0 && channels % ncores != 0) {
                --ncores;
            }
            if (ncores == 0) {
                return std::nullopt;
            }
            const auto core_set = tt::tt_metal::num_cores_to_corerangeset(ncores, grid, true);
            return ShardSpec(core_set, {total_pixels, channels / ncores}, orientation);
        }
        case tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED: {
            uint32_t gy = std::min(static_cast<uint32_t>(grid.y), total_pixels);
            while (gy > 0 && total_pixels % gy != 0) {
                --gy;
            }
            // Same TILE_WIDTH floor on gx for the same reason as WIDTH_SHARDED.
            const uint32_t max_gx_by_alignment = std::max(1u, channels / tt::constants::TILE_WIDTH);
            uint32_t gx = std::min({static_cast<uint32_t>(grid.x), channels, max_gx_by_alignment});
            while (gx > 0 && channels % gx != 0) {
                --gx;
            }
            if (gy == 0 || gx == 0) {
                return std::nullopt;
            }
            CoreRangeSet core_set(
                {tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(gx - 1, gy - 1))});
            return ShardSpec(core_set, {total_pixels / gy, channels / gx}, orientation);
        }
        default: return std::nullopt;
    }
}

}  // namespace

bool override_compatible_with_fast_path(const std::optional<MemoryConfig>& override_mc) {
    // Fast path emits HEIGHT_SHARDED + L1 + ROW_MAJOR (collapsed); reject any other request.
    if (!override_mc.has_value()) {
        return true;
    }
    return override_mc->is_l1() && override_mc->memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED;
}

Fold::program_factory_t Fold::select_program_factory(
    const operation_attributes_t& op_attr, const tensor_args_t& /*tensors*/) {
    // `MultiCore` is RM-only zero-NOC; everything else uses TensorAccessor-based MultiCoreDRAMFold.
    if (op_attr.is_height_sharded_rm_fast_path) {
        return MultiCore{};
    }
    return MultiCoreDRAMFold{};
}

void validate_fold(
    const std::vector<Tensor>& input_tensors,
    bool is_height_sharded_rm_fast_path,
    uint32_t stride_h,
    uint32_t stride_w) {
    const Tensor& input_tensor = input_tensors.at(0);

    const auto& input_shape = input_tensor.padded_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Fold: Expect input tensor to be stored on device.");
    TT_FATAL(input_tensor.buffer() != nullptr, "Fold: Expect input tensor to be allocated on a device buffer.");

    // Mathematical fold constraints — apply to every path.
    TT_FATAL(input_shape[1] % stride_h == 0, "Fold: Input height must be divisible by stride_h.");
    TT_FATAL(input_shape[2] % stride_w == 0, "Fold: Input width must be divisible by stride_w.");

    // Fast path requires the stride_h x stride_w neighbourhood to live on one core (RM L1 sticks).
    // Layout/sharding/buffer-type invariants are already encoded by the flag's construction.
    if (is_height_sharded_rm_fast_path) {
        auto shard_shape = input_tensor.shard_spec().value().shape;
        TT_FATAL(
            shard_shape[0] % (input_shape[2] * stride_h) == 0,
            "Fold (RM fast path): shard height must be divisible by input_width * stride_h.");
        return;
    }

    // Sharded RM (non-fast-path) output rescales shard[0] by sh*sw — needs exact division. Applies
    // uniformly to HEIGHT / WIDTH / BLOCK sharded RM since the rescale formula is the same.
    if (input_tensor.is_sharded() && input_tensor.layout() == Layout::ROW_MAJOR) {
        const auto& shard_shape = input_tensor.shard_spec().value().shape;
        TT_FATAL(
            shard_shape[0] % (stride_h * stride_w) == 0,
            "Fold (sharded RM): shard height ({}) must be divisible by stride_h*stride_w ({}) "
            "so the output shard packs whole folded pixel-rows.",
            shard_shape[0],
            stride_h * stride_w);
    }
}

void Fold::validate_on_program_cache_miss(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    validate_fold({tensors.input_tensor}, op_attr.is_height_sharded_rm_fast_path, op_attr.stride_h, op_attr.stride_w);
}

void Fold::validate_on_program_cache_hit(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    validate_fold({tensors.input_tensor}, op_attr.is_height_sharded_rm_fast_path, op_attr.stride_h, op_attr.stride_w);
}

Fold::spec_return_value_t Fold::compute_output_specs(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    auto input_tensor = tensors.input_tensor;
    const ttnn::Shape& input_shape = input_tensor.logical_shape();
    auto input_dtype = input_tensor.dtype();

    tt::tt_metal::DataType output_dtype =
        (input_dtype == tt::tt_metal::DataType::FLOAT32 ||
         input_dtype == tt::tt_metal::DataType::UINT16)
            ? input_dtype
            : tt::tt_metal::DataType::BFLOAT16;

    // Folded 4D: (N, H/sh, W/sw, C*sh*sw).
    const ttnn::Shape folded_4d_shape(
        {input_shape[0],
         input_shape[1] / op_attr.stride_h,
         input_shape[2] / op_attr.stride_w,
         input_shape[3] * op_attr.stride_h * op_attr.stride_w});

    // Legacy collapsed: (1, 1, N*H/sh*W/sw, C*sh*sw).
    const ttnn::Shape collapsed_shape(
        {1,
         1,
         input_shape[0] * input_shape[1] * input_shape[2] / (op_attr.stride_h * op_attr.stride_w),
         input_shape[3] * op_attr.stride_h * op_attr.stride_w});

    if (op_attr.is_height_sharded_rm_fast_path) {
        // Fast path emits collapsed HEIGHT_SHARDED L1 RM; rescale input spec by sh*sw.
        auto shard_spec = input_tensor.shard_spec().value();
        shard_spec.shape[0] /= op_attr.stride_h * op_attr.stride_w;
        shard_spec.shape[1] *= op_attr.stride_h * op_attr.stride_w;
        // Honor user-supplied shard spec when provided (compatibility already guaranteed by the gate).
        if (op_attr.output_memory_config.has_value() && op_attr.output_memory_config->shard_spec().has_value()) {
            shard_spec = op_attr.output_memory_config->shard_spec().value();
        }
        auto mem_config = MemoryConfig(
            input_tensor.memory_config().memory_layout(), input_tensor.memory_config().buffer_type(), shard_spec);
        return {TensorSpec(
            collapsed_shape,
            tt::tt_metal::TensorLayout(
                output_dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR), mem_config))};
    }

    const bool input_is_tile = input_tensor.layout() == Layout::TILE;
    const ttnn::Shape preserved_4d_shape({input_shape[0], input_shape[1], input_shape[2], input_shape[3]});
    const auto& input_mc = input_tensor.memory_config();
    MemoryConfig output_mc = input_mc;

    if (op_attr.output_memory_config.has_value()) {
        // User-requested output config (mirrors transpose/slice/repeat): honour it.
        const auto& req = op_attr.output_memory_config.value();
        if (req.is_sharded()) {
            std::optional<ShardSpec> spec = req.shard_spec();
            if (!spec.has_value()) {
                spec = generate_fold_shard_spec(input_tensor, folded_4d_shape, req.memory_layout());
            }
            if (spec.has_value()) {
                if (input_is_tile) {
                    // Device op writes preserved_4d; composite reshape rescales by sh*sw — invert here
                    // so the final output's spec matches what the user asked for.
                    auto preserved_spec = spec.value();
                    preserved_spec.shape[0] *= op_attr.stride_h * op_attr.stride_w;
                    preserved_spec.shape[1] /= op_attr.stride_h * op_attr.stride_w;
                    output_mc = MemoryConfig(req.memory_layout(), req.buffer_type(), preserved_spec);
                } else {
                    output_mc = MemoryConfig(req.memory_layout(), req.buffer_type(), spec.value());
                }
            } else {
                // Couldn't synthesise; fall back to interleaved at the requested buffer.
                output_mc = MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, req.buffer_type());
            }
        } else {
            output_mc = req;
        }
    } else if (input_tensor.is_sharded()) {
        // Default universal-IO contract: sharded in → sharded out, same layout/grid/orientation.
        auto out_shard_spec = input_tensor.shard_spec().value();
        if (!input_is_tile) {
            out_shard_spec.shape[0] /= op_attr.stride_h * op_attr.stride_w;
            out_shard_spec.shape[1] *= op_attr.stride_h * op_attr.stride_w;
        }
        output_mc = MemoryConfig(input_mc.memory_layout(), input_mc.buffer_type(), out_shard_spec);
    }

    // Output logical shape: TILE → preserved (composite reshape finalises); sharded RM or
    // user-override or DRAM RM → folded_4d; L1 RM default → legacy collapsed (1,1,X,Y).
    ttnn::Shape output_logical_shape;
    if (input_is_tile) {
        output_logical_shape = preserved_4d_shape;
    } else if (input_tensor.is_sharded() || op_attr.output_memory_config.has_value() || input_mc.is_dram()) {
        output_logical_shape = folded_4d_shape;
    } else {
        output_logical_shape = collapsed_shape;
    }
    return {TensorSpec(
        output_logical_shape,
        tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), output_mc))};
}

Fold::tensor_return_value_t Fold::create_output_tensors(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    return create_device_tensor(compute_output_specs(op_attr, tensors), tensors.input_tensor.device());
}

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::Fold::tensor_return_value_t fold(
    const ttnn::Tensor& input_tensor,
    uint32_t stride_h,
    uint32_t stride_w,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config) {
    using OperationType = ttnn::operations::data_movement::Fold;
    // Fast path needs L1+HEIGHT_SHARDED+ROW_MAJOR input AND a compatible (or default) override.
    const bool input_is_fast_path_compatible =
        input_tensor.is_sharded() && input_tensor.memory_config().is_l1() &&
        input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED &&
        input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR;
    const bool is_height_sharded_rm_fast_path =
        input_is_fast_path_compatible &&
        ttnn::operations::data_movement::override_compatible_with_fast_path(output_memory_config);
    auto operation_attributes = OperationType::operation_attributes_t{
        .stride_h = stride_h,
        .stride_w = stride_w,
        .is_height_sharded_rm_fast_path = is_height_sharded_rm_fast_path,
        .output_memory_config = output_memory_config};
    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
