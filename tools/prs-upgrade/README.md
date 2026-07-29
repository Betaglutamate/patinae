# PRS Upgrade

`prs-upgrade` converts legacy Patinae `.prs` sessions to the current PRS
format. The first supported migration covers sessions written by Patinae
v0.4.0 through v0.4.2, including both raw positional sessions and PRS v2
documents written with named fields.

Run it from the repository root:

```bash
cargo run -p prs-upgrade -- old.prs upgraded.prs
```

The output path must not exist. The tool leaves the source untouched, writes a
current PRS v2 document, and loads the result again before reporting success.
