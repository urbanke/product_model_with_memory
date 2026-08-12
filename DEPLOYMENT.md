# Reproducible compute deployment

Production runs use one Git revision on every machine and the external sealed
table store `tables/anchors_prod`.  Raw corpora, reduced streams, table binaries,
and experiment outputs are intentionally not stored in Git.

The required table metadata for the current production store is:

- `anchors.json`: SHA-256
  `5c409244487deeba8d597ad66fa13479a8d15476d55d4509c33b2ec797b07c68`
- `manifest.json`: SHA-256
  `cf0449a4b704174890f0a1fb95f20954839dd7266ae307e799d1dc6d0234ffb5`

Before a production campaign:

1. pull the same audited Git commit on every host;
2. verify the two hashes above and the presence of all level files;
3. run the production-focused test gate;
4. validate the corpus/reduced-stream manifest and hash;
5. submit through the host's durable scheduler (`nohup` on the M4 only for
   short local work, Slurm on node14 and SCITAS Jed).

Production entry points call `configure_production_tables()` and fail closed
when the sealed store is absent.  Tests requiring the full 2.6 GiB external
store are integration tests and skip in a source-only clone.
