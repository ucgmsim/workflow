# Rupture propagation is not reproducible

`nshm2022-to-realisation` draws each multi-fault event's rupture causality tree
from an **unseeded** random number generator. Two runs of the same rupture, with
the same seeds, produce different trees. This is a bug, not a design choice: the
seed the draw should use is already recorded in every realisation, is already in
the schema, and is already used correctly by a sibling script.

Measured on the 291-event CyberShake NSHM-2022 set, **about 30 events draw a
different causality tree on every run**. No seed carried over from a previous run
can prevent this, because the seed never reaches the generator that matters.

This document records the mechanism, the evidence, the stopgap currently in
place, and what a robust fix looks like. It was written on 2026-08-14 against
`source_modelling` 2026.7.3 and `networkx` 3.6.1.

---

## The mechanism

`workflow/scripts/nshm2022_to_realisation.py:310` seeds numpy, and only numpy:

```python
np.random.seed(seed=seeds.nshm_to_realisation_seed)
```

The module does not import `random` at all. Thirty-seven lines later it calls:

```python
rupture_causality_tree = rupture_propagation.sample_rupture_propagation(...)
```

which reaches, in `source_modelling`:

```
sample_rupture_propagation
  └── sampled_spanning_tree
        └── nx.random_spanning_tree          # @py_random_state(3)
```

`networkx`'s `@py_random_state` decorator means that with `seed=None` — which is
what it is given — `random_spanning_tree` draws from **Python's global `random`
module**. That module auto-seeds itself from OS entropy when it is imported, and
nothing in this code path ever calls `random.seed()`.

So the numpy seed governs the initial-fault choice and the hypocentre, while the
spanning tree is drawn from process entropy.

### The sibling script gets it right

`workflow/scripts/generate_rupture_propagation.py:88` does the correct thing:

```python
random.seed(seeds.rupture_propagation_seed)
np.random.seed(random.randint(0, 2**32 - 1))
```

It seeds Python's `random` from the dedicated seed, then derives numpy's seed
from it. Both generators are deterministic.

`rupture_propagation_seed` is a required field of `SEED_SCHEMA`
(`workflow/schemas.py:971`), is declared on `Seeds`
(`workflow/realisations.py:252`), and is recorded in all 291 deployed
realisations. It exists for exactly this purpose. It is simply never read on the
`nshm2022-to-realisation` path.

---

## Evidence

### Direct reproduction

Real event 322209 (9 faults), seeds taken from its deployed `realisation.json`,
`initial_source` pinned to the deployed root so only the tree draw varies:

```
--- current behaviour: numpy seeded, Python's random not seeded ---
  run 0: differs        run 3: MATCHES deployed
  run 1: differs        run 4: differs
  run 2: differs        run 5: differs
  -> 5 distinct trees across 6 runs

--- with random.seed(seeds.rupture_propagation_seed) added ---
  -> 1 distinct tree across 6 runs
```

Note the second block is stable but still differs from the deployed tree. The
deployed tree came from an unseeded draw; no seed recovers it.

The minimal script that shows this:

```python
import json, random, warnings
from pathlib import Path
import numpy as np
from nshmdb import nshmdb
from source_modelling import rupture_propagation, sources
warnings.filterwarnings("ignore")

RUPTURE = 322209
realisation = json.loads(
    Path(f"/home/arr65/src/cs_nshm_2022/cs_nshm_2022/events/{RUPTURE}/realisation.json").read_text()
)
seeds = realisation["seeds"]
deployed = realisation["rupture_propagation"]["rupture_causality_tree"]
root = next(c for c, p in deployed.items() if p is None)

db = nshmdb.NSHMDB(Path("nshmdb.db"))
faults = {
    name: sources.simplify_fault(f, realisation["srf"]["resolution"])
    for name, f in db.get_rupture_faults(RUPTURE).items()
}

def draw(seed_python: bool) -> frozenset:
    np.random.seed(seed=seeds["nshm_to_realisation_seed"])
    if seed_python:
        random.seed(seeds["rupture_propagation_seed"])
    tree = rupture_propagation.sample_rupture_propagation(
        faults, initial_source=root, strategy="random",
        jump_impossibility_limit_distance=15000,
    )
    return frozenset(frozenset((c, p)) for c, p in tree.items() if p is not None)

print(len({draw(False) for _ in range(6)}), "distinct trees, unseeded")
print(len({draw(True) for _ in range(6)}), "distinct tree, seeded")
```

### How much of the set is affected

A rupture whose pruned jump graph is *already a tree* admits exactly one spanning
tree, so `random_spanning_tree` returns it whatever RNG state it is handed. Only
ruptures whose graph admits more than one are at risk. Counting them with
Kirchhoff's matrix-tree theorem over the real geometry (15 km jump cutoff):

| | events |
|---|---|
| single-fault — no tree to draw | 72 |
| multi-fault, geometry forces one tree | 120 |
| **multi-fault, more than one possible tree** | **99** |

The probability of redrawing a *specific* tree is exact, via the weighted
matrix-tree theorem: `P(T) = ∏w'(e) / Z`, where `w' = w/(1-w)` is the transform
`sampled_spanning_tree` applies and `Z` is the cofactor of the weighted
Laplacian. No sampling is needed.

| P(redraw the deployed tree) | events |
|---|---|
| ≥ 0.999999 | 120 |
| ≥ 0.99 | 131 |
| ≥ 0.9 | 158 |
| ≥ 0.5 | 194 |
| < 0.1 | 3 |

**Expected number of the 219 multi-fault events whose tree differs on any given
run: 29.7.** Least reproducible: `322209` (P = 0.034), `41065` (0.039),
`184263` (0.072). Event `71993` (19 faults) admits 8,883 distinct trees.

The reproduction scripts that produced these tables are
`count_spanning_trees.py` and `tree_probability.py`; both are short enough to
rewrite from the formulas above and neither is checked in.

### Consequences downstream

A different tree changes `jump_points` — and the SRF built from the realisation.
An event whose tree is redrawn has an SRF that no longer corresponds to its
realisation file. So this is not only a provenance problem: it silently
invalidates precomputed products.

---

## Why seed inheritance does not fix it

The campaign inherits seeds from deployed realisations so a regenerated file
replays the original draws. That works for everything numpy governs. It cannot
work here, for two independent reasons:

1. The tree draw never consults any seed, inherited or otherwise.
2. Even once the seeding is fixed, the deployed trees were themselves drawn
   unseeded. The seeded draw is a *fresh* draw from the same distribution — it
   lands on the deployed tree only with probability `P(T)` above. Fixing the
   seeding makes the result **deterministic**, not **equal to the old one**. The
   expected ~30 divergences are the same either way.

There is a related but separate effect worth not confusing with this one:
`9f35c90` changed initial-fault selection from NSHM MFD rates to fault area.
That draw *is* numpy-seeded and so is reproducible, but the distribution changed,
so it can select a different root than the deployed set used. It is a much
smaller effect than the tree draw and is not what makes multi-fault events fail
to reproduce.

---

## The stopgap currently in place

`generate-realisations-from-csv` has an `--inherit-rupture-propagation-from`
option. Given an events directory it copies each event's entire
`rupture_propagation` section — causality tree, jump points and hypocentre —
verbatim over the section the generator derived, after generation succeeds.

```bash
generate-realisations-from-csv nshmdb.db ruptures.csv out/ 24.2.2.1 \
    --inherit-seeds-from                 /path/to/events \
    --inherit-rupture-propagation-from   /path/to/events
```

Inheriting seeds without also inheriting propagation prints a warning, because
that combination looks like reproduction and is not.

**Understand what this buys and what it costs.** It makes the regenerated set
match the deployed set, so the campaign's content verification passes. It does
not make anything reproducible. The section is *copied*, not derived, and the
copy is invisible in the file's own provenance: `log_trail` records the arguments
`nshm2022-to-realisation` was called with, and the carry-over happens outside
that process. A file with an inherited tree is indistinguishable, from the inside,
from one that derived it.

Any run that uses the flag must therefore say so somewhere durable —
`PROVENANCE.md` for the campaign — or the provenance record overstates what was
reproduced.

---

## Fixing it properly

### 1. The minimal fix (a stopgap of its own)

Add one line to `nshm2022_to_realisation.py`, beside the existing numpy seeding:

```python
random.seed(seeds.rupture_propagation_seed)
np.random.seed(seed=seeds.nshm_to_realisation_seed)
```

Add, do not replace: the numpy seeding must keep coming from
`nshm_to_realisation_seed` or every other draw in the script changes.

This makes runs reproducible. Its weakness is that `random.seed()` is
process-global — any other consumer of `random` in the same process, in any
order, perturbs the stream. It is a fix for a script, not for a library.

### 2. The robust fix (preferred; needs `source_modelling`)

Thread an explicit generator through instead of relying on global state:

- `sample_rupture_propagation(..., seed: int | np.random.Generator | None = None)`
- pass it down to `sampled_spanning_tree`
- pass it to `nx.random_spanning_tree(..., seed=seed)` — the parameter already
  exists and `@py_random_state` accepts an int or a `random.Random`

Then `nshm2022_to_realisation` passes `seeds.rupture_propagation_seed`
explicitly, and no global state is involved. This also fixes
`generate_rupture_propagation.py`, whose current correctness depends on nothing
else touching `random` between its `seed()` call and its use.

Audit for the same class of bug while there: any `networkx` function decorated
`@py_random_state` or `@np_random_state` draws from a global unless given
`seed=`.

### 3. Decide what happens to the deployed set

Fixing the seeding does not reproduce the existing 291 realisations — see above.
Whoever picks this up has to choose:

- **Accept the divergence.** Regenerate, let ~30 events land on different trees,
  recompute those events' SRFs. The set becomes genuinely reproducible. This is
  the honest option and the one that matches the campaign's purpose.
- **Keep carrying the section over** with `--inherit-rupture-propagation-from`.
  The set is preserved, but its rupture propagation remains derived from an
  untraceable predecessor for as long as that flag is used.

The first is only expensive because of the SRFs. It is worth checking whether the
affected events' SRFs are actually in use before assuming the cost.

---

## Verifying a fix

1. **Determinism.** Generate the same rupture twice, in separate processes, with
   the same seeds; the `rupture_propagation` sections must be byte-identical. Use
   a rupture with many possible trees — `71993`, `71982`, `71220` and `101449`
   are the widest — because a rupture with one possible tree passes this test
   even when the bug is fully present. That is why the original 9-event sample
   missed it: 8 of the 9 happened to be geometry-forced.
2. **Seed sensitivity.** Changing `rupture_propagation_seed` must change the
   tree. If it does not, the seed still is not reaching the draw.
3. **Coverage.** Run 1 and 2 across a sample of at least 20 multi-fault events
   drawn from the >1-spanning-tree group, not from the set at large — two thirds
   of a random sample cannot detect this bug.

---

## Related

- `docs/superpowers/plans/2026-07-14-traceable-realisation-regeneration.md` — the
  campaign this was found during. Its content verification is what surfaced the
  symptom.
- `verify-realisation-content`'s `values_equivalent` compares numbers on relative
  tolerance with `abs_tol=0.0`. `jump_points` are normalised [0, 1] fault
  coordinates whose true value at a fault edge is zero, delivered by floating
  point as denormal residue — across the deployed set, 387 of 3132 coordinates
  are below 1e-6 and the smallest is 4.3e-216. Relative comparison between two
  such residues is meaningless, and reports a conflict where there is none. That
  is a separate defect in the comparison, not in the science, and it needs a
  per-section absolute tolerance rather than a global one.

---

## Addendum, 2026-08-14: the carry-over is now checked

The stopgap above is no longer a blind copy. Every carried-over section is
compared against the one the pipeline derived anyway — which costs nothing,
because it derives it either way — and the deployed value is adopted **silently
only when the two agree to within the section's tolerance**. That is the case the
carry-over exists for: float noise of a few ULP. Anything larger fails the
rupture until a human records which value to use and why.

```
Skipping rupture 322209: carried-over rupture_propagation differs from the value
derived this run by more than the section tolerance (relative 1e-06, absolute
1e-06) -- .rupture_causality_tree.Kotare - Moutuhora: 'Opotiki 3' != 'Ohae 2'
(+2 more). Record a choice for '322209.rupture_propagation' or
'rupture_propagation' and re-run.
```

Decisions live in a YAML passed as `--inheritance-decisions`, keyed either by
section (applies to every rupture) or `<rupture id>.<section>` (applies to one,
and wins). Each entry needs a `choice` of `inherited`/`derived` and a `reason`;
an entry without a reason is refused at read time.

**This makes the bug visible instead of hiding it.** The events whose causality
tree is redrawn now announce themselves by name on every run, rather than being
silently overwritten with the deployed tree. The campaign still keeps the
deployed value — via one section-wide decision — but does so explicitly, with the
reason recorded next to it.

### The tolerance question this forced

Comparing `rupture_propagation` at all required fixing the relative-only
comparison described under **Related** above. `values_equivalent` now takes an
`abs_tolerance`, defaulting to `0.0` so parameter grids are unchanged, and
`SECTION_TOLERANCES` gives `rupture_propagation` `1e-6` absolute.

That figure is not arbitrary. Jump points are normalised [0, 1] fault
coordinates, so `1e-6` is a millionth of a fault dimension — three orders of
magnitude below the SRF discretisation, far too small to move a single grid
point, and still wide enough to absorb the denormal residue that a relative
comparison inflates into a 100% mismatch. Verified both ways: two residues of the
same fault edge (`1.4e-34` vs `3.2e-07`) compare equal, while a jump point that
actually moved (`0.12` vs `0.83`, the shift seen on event 242445) does not.
