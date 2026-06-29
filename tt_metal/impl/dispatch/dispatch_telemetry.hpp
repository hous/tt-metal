// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <umd/device/types/xy_pair.hpp>

#include <hostdevcommon/dispatch_telemetry_types.hpp>

namespace tt::umd {
class TTDevice;
}  // namespace tt::umd

namespace tt::tt_metal {

/**
 * @brief Read the DispatchCoreTelemetry block from a dispatch core's L1.
 *
 * @param tt_device Non-owning UMD device context.
 * @param noc0_core NOC0 coord of the dispatch core to sample.
 * @return Telemetry data on success, or std::nullopt if the buffer fails signature/version
 *         validation (a warning is logged in that case).
 */
std::optional<DispatchCoreTelemetry> read_dispatch_core_telemetry(tt::umd::TTDevice& tt_device, tt_xy_pair noc0_core);

/**
 * @brief Read the PrefetchCoreTelemetry block from a prefetch core's L1.
 *
 * @param tt_device Non-owning UMD device context.
 * @param noc0_core NOC0 coord of the prefetch core to sample.
 * @return Telemetry data on success, or std::nullopt if the buffer fails signature/version
 *         validation (a warning is logged in that case).
 */
std::optional<PrefetchCoreTelemetry> read_prefetch_core_telemetry(tt::umd::TTDevice& tt_device, tt_xy_pair noc0_core);

}  // namespace tt::tt_metal
