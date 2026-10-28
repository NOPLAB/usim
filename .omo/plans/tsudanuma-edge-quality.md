# Tsudanuma: cleaner observed 3D edge lines

Date: 2026-10-13
Status: execution and final audits recorded on 2026-10-14; quality targets
failed. The high-quality feature remains incomplete. See
`runs/tsudanuma-isaac-local/POINT_EDGE_LINES_RESULTS.md` for executed checks,
fixed measurements, the open native v4 viewer and remaining limitations.
Intent: improve the correctness, continuity and useful coverage of visible
structural edges, not increase the segment count.

## Current evidence

- Frozen input: 12,107,409 observations after iterative LOF k20,
  cutoff 1.2000000476837158, three passes. Island filtering is on hold.
- Current `observed-face-edge-segments-v2`: 237 intersections, zero observed
  boundary lines; default display selects 132 segments.
- The full suite passed 975 tests, but synthetic/regression success does not
  establish real-scene extraction quality.
- `edge_line_geometry.py` fits planes separately in 8 m cores with 0.6 m
  halos, enumerates their intersections and clips finite segments to cores.
  There is no subsequent cross-core consolidation.
- `edge_plane_boundaries.py` revalidates only supplied boundary seeds.
  A missed seed cannot be recovered by that path.
- The surface coherence radius is also 0.6 m. Probes near the local crop
  boundary may have incomplete neighborhoods. This is a cause candidate,
  not yet a measured explanation of the poor result.
- `point-edge-lines-final/qa-error.json` records zero selected lines and
  invisible curves, while the captured image retained the line palette.
  The display defect must be isolated from extraction errors.

## Desired result and constraints

The Isaac viewer should show thin, finite lines along observed building
corners and observed face boundaries, with fewer false lines and fewer
avoidable fragments. It must allow gray source-only, overlay and lines-only
comparison at identical camera poses.

Keep original coordinates, IDs and attributes and the frozen LOF selection.
Do not resume island filtering, refilter source data, connect arbitrary
nearest neighbors, draw hulls or boxes as edges, extrapolate through missing
observations, force Manhattan alignment, or claim complete building outlines.
Observed face boundaries are not automatically physical building edges:
occlusion and incomplete scanning remain separately identified uncertainty.
No commits, pushes, new external dependencies or GPU calculations are
authorized by this plan. Use CPU calculations; any later GPU computation
requires Slurm. Final Isaac display uses the local GPU and stays open.

## Evaluation contract

Freeze six evaluation regions before tuning: a clear building corner, an
observed roof/facade border, an observed opening or facade recess, a sparse
or occluded surface, vegetation/noisy returns, and an edge crossing a core
boundary. Save exact 3D bounds, source IDs, camera poses and reason for
selection in a manifest. Select three for tuning and hold three out;
holdout must include a boundary and a core-crossing example.
Use spatially separate structures for tuning and holdout, with no overlapping
source IDs or surface patches. If one of the six categories is not observable
in the input, record its absence rather than invent a positive example.

Build a small reference from original observations and at least two views,
not from the current output. Store observed reference segments and explicit
negative/ambiguous regions. Record which edge types cannot be established
from these observations. Never count an ambiguous occlusion as a confirmed
physical roof or opening edge. The agent prepares the reference; no mandatory
user annotation is introduced.
Freeze reference uncertainty and inclusion rules before running candidate
methods. Compare algorithm outputs at a common minimum reference length
of 1.5 m. Include short predicted fragments when matching those reference
spans; otherwise fragmentation would masquerade as missing geometry.
Retain references shorter than 1.5 m in a separate diagnostic table. Show
default filtered-display quality separately so hidden lines cannot silently
improve the reported extraction score.

Measure crease and observed-boundary performance separately, as well as
their combined score. Match by length, with 0.10 m positional and 5 degree
direction tolerance; ambiguous regions are reported separately. Precision
is matched predicted length divided by evaluated predicted length. Recall
is covered reference length divided by evaluated reference length.
Deduplicate overlapping predictions before coverage accounting, but report
the duplicate length separately. Also report fragment count per reference,
position error, and unsupported gaps.
Matching requires longitudinal overlap as well as direction and perpendicular
distance; a nearby parallel line outside the reference interval is not a hit.
Each reference interval contributes covered length only once. Count raw
duplicate length and require that consolidation not leave duplicates in the
reference cases. In a region without positive reference edges, report false
positive length directly; do not fabricate a recall or define 0/0 as success.
Give measurement tolerance and reference uncertainty alongside every score.

Targets, not existing achievements: on the held-out unambiguous observations,
precision >= 90%, recall >= 80%, and no invented bridge across a measured gap
larger than 0.25 m. Do not average away a failed edge type or a failed region.
If the input cannot support a target, report the measured limitation rather
than label the feature complete. Counts alone are not quality metrics.
Distinguish model-limited misses (plane budget, minimum area, angle or length
gates) from data-limited misses. Require non-regression against baseline and
the per-type quality targets, not strict precision improvement when baseline
precision is already perfect. Explicitly edge-free controls must contain no
confirmed structural output. Compare original and 4 m shifted tile grids:
covered reference length must agree by at least 95%, with no ownership-only
break or duplicate. Identified holes stay open even when shorter than 0.25 m;
the sampling gap ceiling is not permission to fill an observed opening.

## Todos

- [ ] 1. Establish trustworthy display and a fixed quality baseline.
  - Recommended task executor category: deep-low
  - Paths: `runs/tsudanuma-isaac-local/open_point_edge_lines.py`,
    evaluation scripts and new evidence beneath
    `runs/tsudanuma-isaac-local/edge-quality/`.
  - Diagnose the empty-selection stale render without changing extraction.
    Test nonempty -> empty -> nonempty, type switches, cloud/line visibility,
    and rapid coalesced updates using event-based capture synchronization.
  - Freeze the six-region manifest and observed reference. Capture current
    output at identical poses and calculate the baseline contract above.
  - Acceptance: screenshots agree with selected lines and USD visibility;
    baseline contains omission, false-positive, duplication and fragmentation
    measurements, with ambiguous regions explicitly separated.
  - QA: run the viewer with an isolated evidence directory and inspect every
    native capture. Preserve source IDs and positions. No sleeps or pixel
    retries that merely conceal stale state.

- [ ] 2. Instrument rejection stages and test physical neighborhood completeness.
  - Recommended task executor category: deep-low
  - Paths: `src/usim_env_pipeline/edge_line_geometry.py`,
    `edge_plane_boundaries.py`, `edge_line_support.py`, neighboring tests
    and evaluation scripts.
  - Record counts and reasons for plane rejection, missing boundary seeds,
    angular vacancies, tangent rejection, inadequate face strips, coverage,
    gap splitting, core clipping and minimum length. Retain candidate evidence
    sufficient to explain specific missed reference edges. Include the first
    failed stage per reference span and the acceptance evidence for false
    positives, not just aggregate rejection counts.
  - Instrument the eight-plane budget, 1 m2 area gate, 25 degree dihedral
    gate and display-only suppression. Record boundary proposals consumed
    by unsuccessful finite fits.
  - Compare physical normal radii 0.3, 0.6 and 1.0 m on tuning regions, with
    complete neighborhoods for each setting. Expand the halo or exclude
    incomplete probes; do not interpret truncated neighbors as real edges.
    Derive completeness margins from the actual query radius and the
    projected plane-slab thickness. The current fixed 0.35 m boundary guard
    is invalid for the proposed 0.5 m experiment. Prefer expanding input
    neighborhoods where excluding probes would remove the evaluated edge.
  - Compare plane residuals and density across the reference cases before
    changing 0.05 m plane tolerance or 0.15 m support width.
  - Acceptance: each material miss/false positive in the tuning regions has
    an evidence-backed stage explanation. The radius decision is recorded.
  - QA: deterministic crop/halo and density-variation fixtures; compare
    translated core grids against unchanged reference observations.

- [ ] 3. Correct ownership losses and reconcile cross-core fragments.
  - Recommended task executor category: deep-low
  - Paths: `edge_line_geometry.py`, `edge_line_support.py`,
    `point_edge_lines.py` and corresponding tests.
  - Preserve already supported pre-clip candidates and short ownership
    fragments until reconciliation. Apply final length filtering afterward;
    merging only retained output cannot recover fragments already discarded.
  - Deduplicate overlaps with compatible supporting patches. Merge aligned
    overlapping or touching intervals only after a union refit and fresh
    finite-support validation on original observations. Do not automatically
    join every pair separated by less than the maximum sampling gap.
  - Recompute coverage, endpoints and largest actual gap after clipping and
    after consolidation. Never inherit a pre-clip score as a final score.
  - Avoid rendering one physical corner three times as its crease plus the
    two face boundaries: prefer the confirmed two-face crease for coincident
    output, retain the per-type provenance, and keep nearby distinct lines.
  - Acceptance: the reference core-crossing edge no longer fragments solely
    because of ownership boundaries; nearby distinct parallel edges remain
    separate and >0.25 m unsupported gaps remain split.
  - QA: 0.8 m supported line cut into two 0.4 m fragments; 1.8 m line cut into
    two 0.9 m fragments; shifted tile origins; duplicates across edge types;
    nearby parallel edges; identified holes shorter than 0.25 m; interrupted
    support; endpoint/support consistency. These are synthetic failure cases,
    not assertions that the exact cases occur in Tsudanuma.

- [ ] 4. Improve structural crease candidates where diagnostics justify it.
  - Recommended task executor category: deep-low
  - Paths: `edge_line_geometry.py`, `edge_line_support.py` and neighboring
    tests. Reuse existing plane primitives.
  - If unrelated disconnected inliers explain false lines, compare physical-
    scale surface-patch splitting against the unchanged baseline, including
    sparse but genuinely continuous faces. Do not mandate splitting every
    plane before its quality benefit is established.
  - If confirmed misses arise from plane discovery, area or angle gates,
    change the responsible gate on tuning regions with matched negative
    controls; do not describe those misses as missing source evidence.
  - Retain independent observed strips on both faces and finite support.
    No axis snapping. Adopt only changes that satisfy the evaluation contract
    without baseline regression; keep the unchanged method where it passes.
  - Acceptance: per-type targets and baseline non-regression on tuning
    regions; errors and rejected cases are explained.
  - QA: disconnected coplanar surfaces, sparse continuous faces, small
    observed faces, shallow and oblique creases, noisy corners, continuous
    planes, vegetation-like volume and rigid transforms.

- [ ] 5. Diagnose and recover observed face boundaries.
  - Recommended task executor category: deep-low
  - Paths: `edge_plane_boundaries.py`, `edge_line_geometry.py`,
    `point_edge_lines.py` and corresponding tests.
  - First compare supplied flags and all-face-member proposals on small ROIs,
    holding planes and validators fixed. This isolates proposal starvation
    from failures of angular, tangent or finite-support validation.
  - If proposal starvation is established, generate candidates from observed
    face occupancy and angular vacancies independently of the old mask.
    Compare complete 0.2, 0.3 and 0.5 m boundary neighborhoods.
  - If fitting/consumption is the cause, chain adjacent tangent-compatible
    proposals and fit robust finite straight segments, validating with original
    observations. Split at direction changes and measured gaps. Do not
    replace every stage merely because final boundary count is zero.
  - Keep observed-boundary and two-face-crease provenance distinct; physical
    interpretation remains uncertain where occlusion cannot be excluded.
    Preserve input-mask provenance even if it ceases to be a required gate.
  - Apply the step 3 reconciliation rules to the resulting candidates.
  - Acceptance: per-type targets and baseline non-regression, not compulsory
    recovery of every reference. Zero boundary output fails when confirmed
    boundary references exist. No crop perimeter or scan dropout is labeled
    a confirmed physical building edge.
  - QA: rotated rectangles, recess/opening observations, interior holes,
    crop artifacts, sparse scan dropout and curved boundaries.

- [ ] 6. Evaluate the frozen solution and produce the native comparison.
  - Recommended task executor category: deep-low
  - Paths: evaluation evidence, `docs/point-filter.md`,
    `runs/tsudanuma-isaac-local/POINT_EDGE_LINES_RESULTS.md` and the viewer.
  - Evaluate holdout without further tuning, then process the full frozen LOF
    input. Keep v2 results intact and use a distinct algorithm/cache version.
    Freeze configuration after both candidate branches and reconciliation.
    Include every changed boundary, patch, neighborhood and reconciliation
    parameter in cache identity; boundary function defaults alone are not
    sufficient provenance for tuning.
  - Produce baseline/new metric tables and same-pose source/overlay/lines-only
    images. Show structural creases and observed boundaries separately.
  - Acceptance: meet the evaluation contract or explicitly report failed
    regions, unsupported types and data limitations. Leave the completed
    result open in Isaac; do not use headless output as final visual evidence.
  - QA: related tests, the repository-required full suite, lint/format,
    diagnostics and package build. Record real exit codes and pre-existing
    limitations. Recheck raw IDs, coordinates, cache provenance and native
    curve endpoints.

## Dependencies

1 -> 2 -> 3 -> 4 -> 5 -> 6.
Boundary/crease comparisons are measured separately before consolidation.
Do not run the full-map calculation merely to tune against line count.
Each algorithm change uses tests-before for its observed failure; evaluations
and documentation get no prose-pinning tests.

## Final verification wave

- [ ] F1. Audit source preservation and finite geometric evidence.
  Check frozen-ID hash, original coordinates, per-type witness rules,
  recomputed support scores and no unsupported gap bridging.
- [ ] F2. Audit holdout quality independently of implementation success.
  Inspect every reference region and negative region; confirm per-type
  precision/recall, duplication and fragmentation and list remaining misses.
  Recheck the frozen reference against source-only views, not against new
  predictions; any disputed reference remains explicit uncertainty.
- [ ] F3. Audit the real native viewer and reproducibility.
  Inspect source-only, overlay, lines-only and empty selection at fixed poses,
  validate toggles and leave the final window open. Summarize all executed
  checks and evidence paths.

Commit policy: no workspace commits unless separately requested.

## Planning review

Read-only algorithm review: `st_01a136a9`, completed 2026-10-13.
Its findings were checked against the implementation and incorporated:
short-fragment loss before reconciliation, radius-dependent completeness,
conditional patch splitting, consistent evaluation acceptance, cross-type
duplicates, identified small holes, and complete cache configuration.
The review ran no full-cloud calculation or tests. Pilot quality targets
remain proposed gates, not measured achievements.
