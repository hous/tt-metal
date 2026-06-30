// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_left_shift(const uint shift_amt) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Load/store as two's-complement int32 (sfpi DataLayout::I32 == InstrModLoadStore::INT32),
        // matching the original TTI path. A plain vInt load would read the raw sign-magnitude dest
        // bits and the shift would corrupt the sign bit for negative inputs.
        vInt v = dst_reg[0].mode<sfpi::DataLayout::I32>();
        // WH has no arithmetic shift, so a logical shift must be requested explicitly.
        dst_reg[0].mode<sfpi::DataLayout::I32>() = sfpi::shft(v, static_cast<int>(shift_amt), sfpi::ShiftMode::Logical);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
