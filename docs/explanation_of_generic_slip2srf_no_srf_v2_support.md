  Please examine the source code at /home/arr65/src/EMOD3D/rwg43/current/SlipModel/StandRupFormat/generic_slip2srf.c to determine whether it has any support for writing Version 2 SRF files. If you do find apparent support, please trace through the code to ensure that it is fully implemented 
  
  Verdict

  There is apparent, partial support for Version 2 SRF — a version hook exists and the writer it calls really does have a V2 path — but it is not fully implemented in generic_slip2srf.c. Running this program with version=2.0 would invoke the V2 writer with uninitialized data, producing garbage output or (more
  likely) a segfault.

  The apparent support

  Two things make V2 look supported:

  1. Version is user-settable. It defaults to "1.0" but can be overridden on the command line:
    - generic_slip2srf.c:53 → sprintf(srf.version,"1.0");
    - generic_slip2srf.c:60 → getpar("version","s",srf.version); — so version=2.0 is accepted.
  2. The writer it calls dispatches to a real V2 routine. generic_slip2srf.c:525 calls write_srf(&srf,...), which branches on the version string (srf_subs.c:302):
  if(atof(srf->version) < 2.0)      write_srf1(...);
  else if(atof(srf->version) >= 2.0) write_srf2(...);   // a fully-written function
  2. write_srf2 (srf_subs.c:486) genuinely exists and emits the extra V2 fields.

  Where it falls apart — the trace

  write_srf2's ASCII path (the default, since outbin=0 at generic_slip2srf.c:42) depends on five V2-specific fields that generic_slip2srf.c never populates:

  ┌──────────────────────────────┬─────────────────────────────────────────┬────────────────────────────┐
  │   Field used by write_srf2   │             Where consumed              │ Set in generic_slip2srf.c? │
  ├──────────────────────────────┼─────────────────────────────────────────┼────────────────────────────┤
  │ srf->srf_hcmnt.nline / .cbuf │ header-comment loop, srf_subs.c:560-564 │ No                         │
  ├──────────────────────────────┼─────────────────────────────────────────┼────────────────────────────┤
  │ srf->nseg                    │ drives the POINTS loop, srf_subs.c:586  │ No                         │
  ├──────────────────────────────┼─────────────────────────────────────────┼────────────────────────────┤
  │ srf->np_seg[ig]              │ points-per-segment, srf_subs.c:588      │ No                         │
  ├──────────────────────────────┼─────────────────────────────────────────┼────────────────────────────┤
  │ apval_ptr[i].vs              │ per-point output, srf_subs.c:602        │ No                         │
  ├──────────────────────────────┼─────────────────────────────────────────┼────────────────────────────┤
  │ apval_ptr[i].den             │ per-point output, srf_subs.c:603        │ No                         │
  └──────────────────────────────┴─────────────────────────────────────────┴────────────────────────────┘

  The grep confirms vs, den, the top-level nseg, np_seg, and hcmnt appear nowhere in generic_slip2srf.c. The only srf members it touches are version, type, srf_prect, and srf_apnts (np + apntvals).

  Critically, srf is a plain automatic struct (generic_slip2srf.c:18) with no zero-initialization and no memset, so those unset fields hold indeterminate stack garbage. Concretely, when write_srf2 runs:

  - srf_subs.c:560 — for(i=0;i<srf->srf_hcmnt.nline;i++) loops on an uninitialized nline, then dereferences the uninitialized cbuf pointer.
  - srf_subs.c:586 — for(ig=0;ig<srf->nseg;ig++) is governed by garbage nseg. If it's ≤0, no points are written (empty SRF); if it's >0, line 588's srf->np_seg[ig] dereferences a garbage pointer → segfault.
  - srf_subs.c:602-603 — even in the lucky case, vs/den are written from never-initialized memory.

  ▎ Note: don't be misled by srf.srf_prect.nseg, which is set at generic_slip2srf.c:229. That's the PLANE-rectangle segment count (a different field, used only for the optional PLANE header in both writers). It is not the top-level srf.nseg that drives write_srf2's point output — those are distinct members of
  ▎ struct standrupformat (structure.h:114 vs structure.h:106).

  Secondary observation

  Even write_srf2's binary branch (srf_subs.c:508-549) is essentially a copy of write_srf1's — it iterates apnts_ptr->np directly and omits vs/den, nseg, np_seg, and the header comments. So in binary mode it wouldn't crash, but it also wouldn't produce a genuine V2 file. Either way, generic_slip2srf.c is not
  a working V2 producer.

  Bottom line

  To make generic_slip2srf.c actually write V2 SRF, you'd need to add code (before write_srf at line 525) that: sets srf.nseg and allocates/fills srf.np_seg[], initializes srf.srf_hcmnt (at least nline=0), and assigns each point's vs/den (e.g., from a velocity model). As written, passing version=2.0 is a
  latent crash, not a feature.

