// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Verification and usage example of dispatch telemetry.

#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <sys/utsname.h>

#include <fmt/core.h>

#include <hostdevcommon/common_values.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/experimental/dispatch_telemetry.hpp>

#include "umd/device/tt_device/tt_device.hpp"
#include "umd/device/pcie/pci_device.hpp"
#include "umd/device/arch/wormhole_implementation.hpp"
#include "umd/device/arch/blackhole_implementation.hpp"
#include "umd/device/types/arch.hpp"
#include "umd/device/types/xy_pair.hpp"

namespace {

//--------------------------------

// host_runtime_telemetry.cpp

struct RtTelemetryLoc {
    tt::xy_pair arc_core;  // ARC core NOC coords (NOC0)
    uint64_t addr;         // address to use with read_/write_to_device AND to hand a kernel
    uint32_t size;         // bytes; 0 => telemetry not available
};

// TODO: Should be moved to UMD
inline constexpr uint32_t SCRATCH_RAM_22 = tt::umd::blackhole::ARC_RESET_UNIT_OFFSET + 0x400 + 0x4 * 22;
inline constexpr uint32_t SCRATCH_RAM_23 = tt::umd::blackhole::ARC_RESET_UNIT_OFFSET + 0x400 + 0x4 * 23;

// TTDevice is not an owning context
RtTelemetryLoc discover_runtime_telemetry(tt::umd::TTDevice* dev) {
    RtTelemetryLoc loc{};
    loc.arc_core = dev->get_arc_core();  // arch-agnostic, from UMD

    uint32_t addr32 = 0, size32 = 0;
    // Firmware doc: SCRATCH_RAM[22]=buffer addr, [23]=size in bytes.
    dev->read_from_arc_apb(&addr32, SCRATCH_RAM_22, sizeof(addr32));
    dev->read_from_arc_apb(&size32, SCRATCH_RAM_23, sizeof(size32));
    loc.addr = addr32;  // BH stores an absolute ARC-local NOC addr
    loc.size = size32;
    switch (dev->get_arch()) {
        case tt::ARCH::BLACKHOLE: {
            uint32_t addr32 = 0, size32 = 0;
            // Firmware doc: SCRATCH_RAM[22]=buffer addr, [23]=size in bytes.
            dev->read_from_arc_apb(&addr32, SCRATCH_RAM_22, sizeof(addr32));
            dev->read_from_arc_apb(&size32, SCRATCH_RAM_23, sizeof(size32));
            loc.addr = addr32;  // BH stores an absolute ARC-local NOC addr
            loc.size = size32;
            break;
        }
#if 0
        case tt::ARCH::WORMHOLE_B0: {
            // TODO: For wormhole additions, CONFIRM the slot indices
            // against your WH firmware before trusting it. WH stores a CSM-local *offset*,
            // so add the ARC CSM NOC base.
            uint32_t off32 = 0, size32 = 0;
            dev->read_from_arc_apb(&off32,  SCRATCH_RAM_22, sizeof(off32)); // verify offset
            dev->read_from_arc_apb(&size32, SCRATCH_RAM_23, sizeof(size32));
            loc.addr = static_cast<uint64_t>(off32) + tt::umd::wormhole::ARC_CSM_OFFSET_NOC; // 0x810000000
            loc.size = size32;
            break;
        }
#endif
        default: UMD_THROW(tt::umd::error::RuntimeError, "runtime telemetry not supported on this arch");
    }
    return loc;
}

// Host read / write

// Reads `n` bytes from the buffer into `dst`. Returns bytes available (0 = unavailable).
uint32_t host_read_runtime_telemetry(tt::umd::TTDevice* dev, void* dst, uint32_t n) {
    RtTelemetryLoc loc = discover_runtime_telemetry(dev);
    if (loc.size == 0) {
        return 0;
    }
    uint32_t to_read = std::min(n, loc.size);
    dev->read_from_device(dst, loc.arc_core, loc.addr, to_read);  // PCIe->NOC, synchronous
    return to_read;
}

// Writes `n` bytes from `src` into the buffer (bounded by its size). Returns bytes written.
uint32_t host_write_runtime_telemetry(tt::umd::TTDevice* dev, const void* src, uint32_t n) {
    RtTelemetryLoc loc = discover_runtime_telemetry(dev);
    if (loc.size == 0) {
        return 0;
    }
    uint32_t to_write = std::min(n, loc.size);
    dev->write_to_device(src, loc.arc_core, loc.addr, to_write);
    return to_write;
}

std::string format_bytes(const std::vector<uint8_t>& data) {
    std::string result;
    for (size_t i = 0; i < data.size(); ++i) {
        if (i != 0) {
            result += " ";
        }
        result += fmt::format("{:02x}", data[i]);
    }
    return result;
}

bool test_host_runtime_telemetry(tt::umd::TTDevice* dev) {
    fmt::print("Runtime telemetry host write/read test: discovering buffer.\n");
    const RtTelemetryLoc loc = discover_runtime_telemetry(dev);
    fmt::print(
        "Runtime telemetry buffer: arc_core=({}, {}) addr=0x{:x} size={} bytes.\n",
        loc.arc_core.x,
        loc.arc_core.y,
        loc.addr,
        loc.size);

    if (loc.size == 0) {
        fmt::print(stderr, "Runtime telemetry buffer is unavailable; skipping host write/read test.\n");
        return false;
    }

    const uint32_t test_size = std::min<uint32_t>(loc.size, 16);
    std::vector<uint8_t> original(test_size);
    std::vector<uint8_t> test_buffer(test_size);
    std::vector<uint8_t> readback(test_size, 0);

    for (uint32_t i = 0; i < test_size; ++i) {
        test_buffer[i] = static_cast<uint8_t>(0xa0 + i);
    }

    fmt::print("Step 1: reading {} original byte(s) from runtime telemetry buffer.\n", test_size);
    const uint32_t original_read = host_read_runtime_telemetry(dev, original.data(), test_size);
    fmt::print("Step 1: read {} byte(s): {}\n", original_read, format_bytes(original));

    fmt::print("Step 2: writing test buffer: {}\n", format_bytes(test_buffer));
    const uint32_t written = host_write_runtime_telemetry(dev, test_buffer.data(), test_size);
    fmt::print("Step 2: wrote {} byte(s).\n", written);

    fmt::print("Step 3: reading back test buffer.\n");
    const uint32_t read = host_read_runtime_telemetry(dev, readback.data(), test_size);
    fmt::print("Step 3: read {} byte(s): {}\n", read, format_bytes(readback));

    const bool match = written == test_size && read == test_size && readback == test_buffer;
    fmt::print("Step 4: compare readback to written buffer: {}.\n", match ? "PASS" : "FAIL");

    fmt::print("Step 5: restoring original runtime telemetry bytes.\n");
    const uint32_t restored = host_write_runtime_telemetry(dev, original.data(), original_read);
    fmt::print("Step 5: restored {} byte(s).\n", restored);

    return match;
}

#if 0
//To let a kernel use the buffer, hand it the discovered values as runtime args:
RtTelemetryLoc loc = discover_runtime_telemetry(dev);
// arg layout the device functions below expect: [arc_x, arc_y, csm_addr, size, l1_scratch]
tt::tt_metal::SetRuntimeArgs(program, kernel, cores,
    {(uint32_t)loc.arc_core.x, (uint32_t)loc.arc_core.y,
     (uint32_t)loc.addr, loc.size, /*l1 scratch addr*/ L1_SCRATCH});
// (On Blackhole loc.addr fits in 32 bits — CSM-local addresses are in the 0x1000_0000 range. On Wormhole the value exceeds 32 bits, so the kernel path below needs get_noc_addr_helper with the full 64-bit address rather than get_noc_addr; another reason the verified path here is Blackhole.)

// Device read / write (kernel)
// These are inherently arch-agnostic: all arch specifics were resolved on the host and passed in as args.

// device_runtime_telemetry.cpp  (compiled as a data-movement kernel)
#include "dataflow_api.h"

// Reads the whole buffer from ARC CSM into local L1 scratch.
void device_read_runtime_telemetry() {
    uint32_t arc_x   = get_arg_val<uint32_t>(0);
    uint32_t arc_y   = get_arg_val<uint32_t>(1);
    uint32_t csm_addr= get_arg_val<uint32_t>(2);   // ARC-local NOC addr (32-bit on BH)
    uint32_t size    = get_arg_val<uint32_t>(3);
    uint32_t l1_dst  = get_arg_val<uint32_t>(4);   // local L1 scratch (>=16B aligned)

    uint64_t src = get_noc_addr(arc_x, arc_y, csm_addr);  // uses noc_index (run kernel on NOC0)
    noc_async_read(src, l1_dst, size);
    noc_async_read_barrier();                              // data valid in L1 after this
    invalidate_l1_cache();                                 // cached-L1 parts (Quasar) before CPU read
}

// Writes local L1 scratch back into the ARC CSM buffer.
void device_write_runtime_telemetry() {
    uint32_t arc_x   = get_arg_val<uint32_t>(0);
    uint32_t arc_y   = get_arg_val<uint32_t>(1);
    uint32_t csm_addr= get_arg_val<uint32_t>(2);
    uint32_t size    = get_arg_val<uint32_t>(3);
    uint32_t l1_src  = get_arg_val<uint32_t>(4);

    uint64_t dst = get_noc_addr(arc_x, arc_y, csm_addr);
    noc_async_write(l1_src, dst, size);
    noc_async_write_barrier();                             // write retired at the ARC endpoint

    // For a shared counter instead of a bulk write, prefer the atomic (no RMW race):
    //   noc_semaphore_inc(get_noc_addr(arc_x, arc_y, counter_addr), 1);
    //   noc_async_atomic_barrier();
}
#endif
//--------------------------------

struct KernelVersion {
    unsigned major = 0;
    unsigned minor = 0;
    std::string release;
};

bool get_linux_kernel_version(KernelVersion& version) {
    utsname info{};
    if (uname(&info) != 0) {
        return false;
    }

    version.release = info.release;
    char* end = nullptr;
    version.major = static_cast<unsigned>(std::strtoul(version.release.c_str(), &end, 10));
    if (end == version.release.c_str() || *end != '.') {
        return false;
    }

    const char* minor_start = end + 1;
    version.minor = static_cast<unsigned>(std::strtoul(minor_start, &end, 10));
    return end != minor_start;
}

bool require_supported_kernel() {
    KernelVersion version;
    if (!get_linux_kernel_version(version)) {
        fmt::print(stderr, "Unable to determine Linux kernel version; refusing to run dispatch_telemetry_dump.\n");
        return false;
    }

    if (version.major > 5 || (version.major == 5 && version.minor >= 15)) {
        return true;
    }

    fmt::print(
        stderr,
        "dispatch_telemetry_dump requires Linux kernel 5.15 or newer; found {}.\n"
        "tt-kmd memory.c::is_pin_pages_size_safe() documents that with IOMMU enabled on Linux 5.4,\n"
        "large page pinnings can soft-lock during unpin in:\n"
        "  tt_cdev_release/ioctl_unpin_pages -> unmap_sg -> __unmap_single -> iommu_unmap_page\n",
        version.release);
    return false;
}

[[maybe_unused]] bool print_snapshot(tt::tt_metal::IDevice* device, tt::tt_metal::DispatchTelemetry& telemetry) {
    auto info = telemetry.read_info();
    fmt::print(
        "dispatch_telemetry_dump  chip={}  num_hw_cqs={}  telemetry_api_version={}  ts={}s\n",
        device->id(),
        device->num_hw_cqs(),
        telemetry.version(),
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count());

    if (!info.has_value()) {
        fmt::print(stderr, "Failed to read dispatch telemetry info; see log warnings for validation details.\n");
        return false;
    }

    fmt::print(
        "{:>3} {:<9} {:>8} {:>26} {:>24}\n",
        "cq",
        "component",
        "waiting",
        "blocked_since_last_read",
        "work_since_last_read");
    for (const auto& cq_info : info->info_cqs) {
        fmt::print(
            "{:>3} {:<9} {:>8} {:>26} {:>24}\n",
            cq_info.cq_id,
            "prefetch",
            cq_info.prefetch_waiting_on_upstream ? "yes" : "no",
            cq_info.prefetch_blocked_count_since_last_read,
            cq_info.prefetch_command_count_since_last_read);
        fmt::print(
            "{:>3} {:<9} {:>8} {:>26} {:>24}\n",
            cq_info.cq_id,
            "dispatch",
            cq_info.dispatch_waiting_on_upstream ? "yes" : "no",
            cq_info.dispatch_blocked_count_since_last_read,
            cq_info.program_count_since_last_read);
    }
    std::cout.flush();
    return true;
}

[[maybe_unused]] tt::tt_metal::Program create_blank_program(const tt::tt_metal::CoreCoord& core) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    tt::tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
    return program;
}

[[maybe_unused]] void issue_workloads(tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t num_programs) {
    if (num_programs == 0) {
        return;
    }

    auto& cq = mesh_device->mesh_command_queue();
    const auto target_devices = tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());
    const tt::tt_metal::CoreCoord worker_core{0, 0};

    fmt::print("Issuing {} blank workload program(s).\n", num_programs);
    for (uint32_t i = 0; i < num_programs; ++i) {
        tt::tt_metal::distributed::MeshWorkload workload;
        workload.add_program(target_devices, create_blank_program(worker_core));
        tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
    }
    tt::tt_metal::distributed::Finish(cq);
}

}  // namespace

int main(int argc, char** argv) {
    if (!require_supported_kernel()) {
        return 1;
    }

    uint32_t device_index = 0;
    bool list_devices = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--device" || a == "-d") && i + 1 < argc) {
            const long value = std::atol(argv[++i]);
            device_index = value > 0 ? static_cast<uint32_t>(value) : 0;
        } else if (a == "--list-devices") {
            list_devices = true;
        } else if (a == "--help" || a == "-h") {
            fmt::print("Usage: {} [--device INDEX] [--list-devices]\n", argv[0]);
            return 0;
        }
    }

    const std::vector<int> pci_device_ids = tt::umd::PCIDevice::enumerate_devices();
    if (pci_device_ids.empty()) {
        fmt::print(stderr, "No PCIe devices found.\n");
        return 1;
    }
    if (list_devices) {
        fmt::print("Enumerated UMD PCIe devices:\n");
        for (size_t i = 0; i < pci_device_ids.size(); ++i) {
            fmt::print("  index {} -> PCIe device id {}\n", i, pci_device_ids[i]);
        }
    }
    if (device_index >= pci_device_ids.size()) {
        fmt::print(
            stderr,
            "Requested device index {} but only {} PCIe device(s) were found.\n",
            device_index,
            pci_device_ids.size());
        return 1;
    }

    const int pci_device_id = pci_device_ids.at(device_index);
    fmt::print("Creating UMD TTDevice handle for PCIe device id {} at index {}.\n", pci_device_id, device_index);
    std::unique_ptr<tt::umd::TTDevice> tt_device = tt::umd::TTDevice::create(pci_device_id);

    const bool runtime_telemetry_test_passed = test_host_runtime_telemetry(tt_device.get());

    if (!runtime_telemetry_test_passed) {
        return 1;
    }

#if 0
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(
        device_index,
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1,
        tt::tt_metal::DispatchCoreConfig{tt::tt_metal::DispatchCoreType::WORKER});
    tt::tt_metal::IDevice* device = mesh_device->get_devices().front();

    tt::tt_metal::DispatchTelemetry telemetry(*device);
    if (!telemetry.read_info().has_value()) {
        fmt::print(
            stderr, "Failed to read initial dispatch telemetry info; see log warnings for validation details.\n");
        return 1;
    }

    // TODO: Inspect device while it runs an independent workload instead of launching our own
    //       once the SMC/ARC region supports sharing dispatch core locations.
    issue_workloads(mesh_device.get(), num_programs);

    if (!print_snapshot(device, telemetry)) {
        return 1;
    }
#endif
    return 0;
}
