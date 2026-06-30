// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: DFB self-loop loopback on a single DM kernel.
// One DM engine both fills and drains the same DFB (bound as PRODUCER and CONSUMER under a
// shared self-loop-pair accessor name dfb::scratch). This is the classic Gen1 scratch pattern:
// reserve -> read DRAM into the DFB slot -> push -> wait -> write the slot back to DRAM -> pop.
// On Gen1 a DFB lowers to a plain circular buffer, so the sequential push-then-pop on one RISC
// is well-defined. (On Gen2 such a self-loop is rejected at spec validation.)
//
// Runtime args:
//   arg 0: source DRAM address
//   arg 1: destination DRAM address
//   arg 2: DRAM bank ID
//   arg 3: number of entries to transfer

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t bank_id = get_arg_val<uint32_t>(2);
    uint32_t num_entries = get_arg_val<uint32_t>(3);

    // Same underlying DFB on both endpoints (self-loop pair) -> one dfb::scratch token.
    DataflowBuffer buf(dfb::scratch);
    uint32_t entry_size = buf.get_entry_size();

    for (uint32_t i = 0; i < num_entries; i++) {
        // Fill: DRAM -> DFB slot.
        buf.reserve_back(1);
        uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, src_addr);
        noc_async_read(src_noc_addr, buf.get_write_ptr(), entry_size);
        noc_async_read_barrier();
        buf.push_back(1);

        // Drain: DFB slot -> DRAM.
        buf.wait_front(1);
        uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, dst_addr);
        noc_async_write(buf.get_read_ptr(), dst_noc_addr, entry_size);
        noc_async_write_barrier();
        buf.pop_front(1);

        src_addr += entry_size;
        dst_addr += entry_size;
    }
}
