# Paper vs. Code Gap Analysis (2026-03-29)

**Executive Summary:** The paper describes a package-aware modeling framework for heterogeneous GPU + accelerator systems. The code implements the heterogeneous GPU + accelerator infrastructure and parameterizable accelerator interconnect, but is missing **two critical components** to achieve the research goal: parameterizable GPU-to-DRAM interconnect and chiplet topology modeling. Additionally, the paper makes several overclaimed assertions (thermal modeling, PPA analysis) not implemented in code, and Section II (Detailed Plan) contradicts Section III (actual implementation).

---

## 1. Critical Gaps

### 1.1 GPU-to-DRAM Interconnect Not Parameterizable (RESEARCH GOAL BLOCKER)

**Paper (Section III-A):**
> "Within each GPU, internal communication between the command processor, compute units, L1 caches, L2 cache, and DRAM controller uses zero-latency direct connections—modeling the on-die wiring within a single chiplet."

**Code Reality:**
- GPU's L2 → DRAM path: `DirectConnection` (from [r9nano/builder.go:272-301](../amd/samples/runner/timingconfig/r9nano/builder.go))
- DRAM controller: Fixed 100-cycle latency (hardcoded in [r9nano/builder.go:507-525](../amd/samples/runner/timingconfig/r9nano/builder.go))
- No bandwidth modeling, no contention between concurrent L2 banks

**Problem:**
DRAM (HBM) is **not** on-die — it sits on a separate stack connected via interposer/TSVs. This is a _physical link_ with real latency and bandwidth constraints, exactly like the accelerator's internal PCIe. However:
- Accelerator: Internal PCIe has bandwidth + latency parameters (`--accel-interconnect-bw`, `--accel-interconnect-latency`)
- GPU: Zero-latency link, no parameters

**Impact on Research Goal:**
The goal is **"package-aware modeling of the full heterogeneous system."** This means you should be able to study how packaging technology affects _all_ IPs uniformly. Without a parameterizable GPU→DRAM link, you **cannot**:
- Study GPU packaging effects (e.g., different interposer technologies)
- Compare how GPU-side and accelerator-side packaging choices interact
- Make meaningful architectural decisions about where to place GPU vs accelerator dies
- The GPU becomes a second-class citizen vs the accelerator

**What's Needed:**
Replace the GPU's `L2ToDRAMConn` `DirectConnection` with a bandwidth-aware network similar to the accelerator's internal PCIe. This should:
- Use `pcie.NewConnector()` or similar [Akita](https://github.com/sarchlab/akita) network component
- Have CLI flags for bandwidth and latency (e.g., `--gpu-dram-bw`, `--gpu-dram-latency`)
- Be parameterizable per GPU or globally
- Allow sweeping different technologies (electrical interposer vs optical, etc.)

**Effort:** Medium (following the accelbuilder pattern from [amd/samples/runner/timingconfig/accelbuilder/builder.go](../amd/samples/runner/timingconfig/accelbuilder/builder.go))

---

### 1.2 Chiplet Topology Not Implemented (RESEARCH GOAL BLOCKER)

**Paper (Section III-D, lines 269–336):** Describes a fully-designed 3-tier hierarchical network:
- **Tier 1 (inter-chiplet):** Package substrate / optical bridge, configurable BW + latency
- **Tier 2 (intra-chiplet):** One network per chiplet, different TBW + latency per chiplet
- **Tier 3 (accel-to-DRAM):** Existing internal PCIe (already implemented)

With `chiplet_config.json` specifying device placement, routing traffic through the appropriate tier, and enabling studies like:
- Co-locating GPU + accelerator on same chiplet (intra-chiplet latency) vs separate chiplets (inter-chiplet latency)
- Scaling studies (1, 2, 4 chiplets)
- Optical vs electrical at different tiers

**Code Reality:**
- Zero implementation
- No `chipletbuilder/` package exists
- No `chiplet_config.json` parsing
- No `--chiplet-config` CLI flag
- All devices connect through a **flat** PCIe topology (GPU pairs under shared switch, each accel under its own switch)

The design document ([docs/chiplet_design.md](./chiplet_design.md)) is complete and detailed, but no Go code has been written.

**Impact on Research Goal:**
This is the _main_ contribution of the framework. Without it, you cannot study:
- Device placement effects on heterogeneous system performance
- Topology/packaging architecture designs
- The "first-class" research question: "How should I physically arrange GPU, accelerator, and memory in a package?"

**What's Needed:**
Implement the three-tier hierarchical network builder as described in [chiplet_design.md](./chiplet_design.md):
1. Create `amd/samples/runner/timingconfig/chipletbuilder/` package
2. Parse `chiplet_config.json` (schema defined in chiplet_design.md section 4)
3. Create three independent `pcie.Connector` instances (one per tier)
4. Route device ports through appropriate tier based on chiplet assignment
5. Wire inter-chiplet gateways
6. Add metrics for intra/inter-chiplet traffic

**Effort:** Medium–Large (30–40% of a typical research project)

---

## 2. Discrepancies in Paper Claims

### 2.1 Section II Describes gem5, Section III Describes MGPUSim

**Paper Section II (Detailed Plan of Work, lines 5–23):**
> "The primary objective of this phase is to construct a unified simulation environment ... At the foundation of this infrastructure is **gem5**, an open-source, discrete-event architectural simulation platform. To accurately model domain-specific hardware accelerators (NPUs), we leverage **gem5-SALAM** (System Architecture for LLVM-based Accelerator Modeling)... Following the methodology ... we will establish a full-system SoC environment."

> "The second phase involves... extend **gem5's Ruby memory system** and **Garnet Network-on-Chip (NoC) models**..."

**Paper Section III-A (Framework Design, line 1–4):**
> "This section describes the heterogeneous simulation infrastructure we have built on top of **MGPUSim**, a cycle-accurate GPU simulator based on the **Akita discrete-event simulation engine**."

**Code Reality:**
Entire implementation is **MGPUSim + Akita**, not gem5. The accelerator uses:
- Host CPU functional execution (Python fallback pattern)
- Analytical timing model (compute cycles based on PE array dimensions)
- Full-traffic DMA through Akita PCIe network
- NOT LLVM IR execution

**Analysis:**
Section II is the original proposal plan. Section III is the actual implementation. They contradict each other. Section II must have been written before the pivot to MGPUSim was finalized.

**What's Needed:**
Rewrite Section II to reflect the actual MGPUSim + Akita implementation, OR remove Section II entirely and replace with "Implementation Approach" that explains the pivot and rationale.

**Impact:** Confusion for reviewers; makes paper seem incomplete.

---

### 2.2 SRAM (32 MB) Shown in Diagram but Not Functional

**Paper (Section III, Fig. 2 and text line 122–123):**
> "Accelerator 0 (detailed) — center x = 3.2"
> Shows: PE Array (256×256) → SRAM (32 MB) → **PCIe Interconnect** → DRAM Ctrl

**Code Reality:**
- [builder.go:17](../amd/timing/accelerator/builder.go#L17): `sramSizeBytes uint64` field defined
- [builder.go:29](../amd/timing/accelerator/builder.go#L29): Default set to `32 * 1024 * 1024`
- [builder.go:87](../amd/timing/accelerator/builder.go#L87): Passed to accelerator component
- [comp.go:91](../amd/timing/accelerator/comp.go#L91): Stored in component
- **Never read anywhere** — no tiling logic, no capacity checks, no SRAM-to-DRAM spillover

Data flow: AccelUnit → ToMem port → internal PCIe → DRAM (SRAM is bypassed)

**Analysis:**
The diagram implies SRAM is part of the active data path. In reality, it's a dead parameter. This is documented as future work in CLAUDE.md ("NOT yet modeled: SRAM capacity/tiling, double-buffering").

**What's Needed:**
Either:
1. Remove SRAM from Fig. 2, OR
2. Add a footnote to Fig. 2 cap noting that SRAM capacity is not yet modeled (Phase 4 future work)

---

### 2.3 Thermal and Multi-Physics Modeling Claimed but Not Implemented

**Paper Abstract (lines 2–4):**
> "By integrating physical packaging constraints - specifically **thermal-aware latency** and **interconnect density profiles** into established simulation infrastructures..."

**Paper Introduction (lines 7–8):**
> "the choice of packaging technology ... directly shapes the achievable bandwidth, latency, power efficiency, **and thermal envelope** of the resulting system."

> "how a **thermal hotspot on one die increases wire resistance** on the interposer"

**Code Reality:**
Zero thermal code anywhere:
- No temperature tracking
- No dynamic frequency throttling
- No thermal-dependent latency
- No power modeling
- No thermal coupling between dies
- Not integrated with HotSpot or similar

**Analysis:**
The paper makes these claims to motivate the problem, but they are not implemented features. The framework models **static bandwidth and latency**, not dynamic thermal effects.

**What's Needed:**
Clarify in abstract/intro that the initial version focuses on **interconnect technology** (BW + latency). Thermal modeling is _future work_ (Phase 4), not part of current deliverable. OR integrate a power + thermal model (significant work).

**Impact:** Minor. The interconnect-only studies are still valuable, but reviewers may expect thermal support based on abstract language.

---

### 2.4 Power and Area Analysis Claimed but Not Implemented

**Paper Abstract & Section III Phase III:**
Multiple references to "Performance-Power-Area (PPA) trade-offs" and "PPA evaluation."

**Code Reality:**
Only **performance** metrics are implemented:
- Execution time (busy time)
- Memory traffic (read/write bytes, request counts)
- Utilization (accelerator busy cycles)
- NO power estimation
- NO area estimation

**Analysis:**
PPA is a standard design optimization goal, but the framework only addresses the "P" part. Power and area are complex topics requiring additional models.

**What's Needed:**
Scope paper to "Performance trade-offs" or "Interconnect performance impact" rather than PPA. OR add analytical power model (e.g., based on traffic estimates).

**Impact:** Minor. Performance insights are still valuable.

---

## 3. Accurate Descriptions

These aspects of the paper **correctly** describe the code:

| Aspect | Paper Reference | Code Match | Status |
|--------|-----------------|-----------|--------|
| Unified global storage | Section III-A-2 | `globalStorage` in builder.go | ✓ |
| Address partitioning | Section III-A-2 | `memAddrOffset` per device | ✓ |
| Per-device DRAM controllers | Section III-A-2 | 16 per GPU, 1 per accel | ✓ |
| Accelerator full-traffic DMA | Section III-A-4 | 64-byte requests in comp.go | ✓ |
| DeviceTensor interface | Section III-B | `tensor.DeviceTensor` + `Ptr()` | ✓ |
| Layer-wise assignment (accel_config.json) | Section III-B | Implemented in all benchmarks | ✓ |
| Dual timing model | Section III-C | GPU cycle-accurate, accel analytical | ✓ |
| READ→COMPUTE→WRITE phases | Section III-C | Explicit state machine in comp.go | ✓ |
| Parameterizable accel interconnect | Section III-C | CLI flags, Akita PCIe connector | ✓ |
| Multi-GPU ring all-reduce | Section III-B-3 | `mccl/allreduce.go` | ✓ |
| System topology overview | Section III-A + Fig. 2 | Matches code (flat PCIe tree) | ✓ (except chiplet) |

---

## 4. Priority Ranking for Fixes

| Priority | Item | What | Where | Effort | Impact on Research |
|----------|------|------|-------|--------|------------------|
| **P0** | Parameterize GPU→DRAM interconnect | Replace `DirectConnection` with PCIe connector | `r9nano/builder.go` lines 272–301 | Medium | **CRITICAL** — enables symmetric packaging studies |
| **P0** | Implement chiplet topology | Build `chipletbuilder/`, parse JSON, create 3-tier network | New package `amd/samples/runner/timingconfig/chipletbuilder/` | Medium–Large | **CRITICAL** — main research contribution |
| **P1** | Rewrite Section II | Update to reflect MGPUSim reality or remove | `pkg_awr_overleaf/sections/02_detailed_plan_of_work.tex` | Low | Clarity for reviewers |
| **P1** | Scope thermal/PPA claims | Remove or mark as future work | Abstract and intro | Low | Honesty in framing |
| **P2** | Annotate SRAM | Add footnote to Fig. 2 explaining why SRAM is dead parameter | `pkg_awr_overleaf/sections/04_design.tex` line 122–123 | Low | Accuracy |
| **P2** | Document system-level PCIe | Explain why it's not the focus of interconnect studies | Paper section III | Low | Completeness |
| **P3** | Make DRAM latency configurable | Add CLI flag instead of hardcoding 100 cycles | `builder.go` and `accelbuilder/builder.go` | Low | Nice-to-have for experiments |

---

## 5. What Must Be Done Before Paper Submission

### Blocking Issues (P0)
1. **Implement chiplet topology** — It's a core claim in Section III. Reviewers will expect code.
2. **Implement parameterizable GPU→DRAM** — Without it, research goal is not achieved.

### Should Fix Before Submission (P1)
1. Rewrite Section II to match implementation
2. Scope thermal/PPA claims appropriately

### Nice-to-Have (P2–P3)
- Annotate SRAM
- Explain system PCIe
- Make DRAM latency configurable

---

## 6. Summary Table: Paper Promises vs. Code Delivery

| Feature | Paper | Code | Status |
|---------|-------|------|--------|
| Heterogeneous GPU + accelerator simulation | ✓ Described | ✓ Implemented | Ready |
| Cycle-accurate GPU timing | ✓ Described | ✓ Implemented (MGPUSim) | Ready |
| Analytical accelerator timing | ✓ Described | ✓ Implemented | Ready |
| Full-traffic DMA simulation for accelerator | ✓ Described | ✓ Implemented | Ready |
| Parameterizable accelerator interconnect | ✓ Described | ✓ Implemented (CLI flags) | Ready |
| Multi-GPU support | ✓ Described | ✓ Implemented (ring all-reduce) | Ready |
| GPU-to-DRAM parameterizable interconnect | ⚠ Mentioned (on-die wiring) | ✗ Not implemented | **Blocking** |
| **Chiplet topology (3-tier hierarchy)** | ✓ Described (Section III-D) | ✗ Not implemented (design doc only) | **Blocking** |
| Thermal modeling | ⚠ Claimed in abstract | ✗ Not implemented | Overclaimed |
| Power modeling | ⚠ Claimed in abstract | ✗ Not implemented | Overclaimed |
| Area modeling | ⚠ Claimed in abstract | ✗ Not implemented | Overclaimed |
| gem5-SALAM foundation | ✓ Described (Section II) | ✗ Uses MGPUSim instead | Conflicting |

---

## Appendix: Research Goal vs. Current Capabilities

**Stated Research Goal (abstract + intro):**
> "Package-aware modeling framework designed to bridge [the divide between logical and physical soundness]. By integrating physical packaging constraints into simulation, we enable architects to program the top-level architecture by defining chiplet placement and interconnect strategies, then evaluate system using cycle-accurate data-flow analysis."

**Current Capabilities:**
- ✓ Heterogeneous GP + accelerator simulation
- ✓ Cycle-accurate GPU side
- ✓ Parameterizable accelerator interconnect (one link)
- ✓ Multiple GPU/accelerator instances
- ✗ **GPU packaging effects** (no parameterizable GPU→DRAM link)
- ✗ **Chiplet topology awareness** (no hierarchical network, no placement modeling)
- ✗ **Thermal-aware modeling** (zero thermal code)
- ✗ **Power or area analysis** (zero power/area code)

**Gap:** ~50% of research goal is implemented. Chiplet topology + GPU→DRAM parameterization are **critical** for the remaining 50%.

