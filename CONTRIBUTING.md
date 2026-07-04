# Contributing to the OpenBiomechanics Project

Thanks for your interest in improving OBP.

## Collaboration & data licensing

For research collaboration, commercial data licensing, or questions about the
larger IRB-approved dataset, email **sportsscience@drivelinebaseball.com** and
CC **gretchen@drivelinebaseball.com**.

## Working with the code and data

- **Get the data:** it is distributed via [GitHub Releases](https://github.com/drivelineresearch/openbiomechanics/releases),
  not committed to git. Run `scripts/download_data.sh` (requires the `gh` CLI).
- **Dependencies** are pinned per area in `requirements.txt`
  (`computer_vision/` and each `<module>/code/`).
- **Orientation:** read [`CLAUDE.md`](CLAUDE.md) for the repository map, the
  join keys, and the conventions.

## Pull requests

1. Branch from `main`, one focused change per branch.
2. **Do not commit large binaries** (raw C3D, `full_sig` zips, video,
   installers). `.gitignore` blocks them; large artifacts belong in a Release.
3. If you touch a README's POI or `metadata.csv` dictionary, verify the entries
   against the actual CSV headers.
4. Code is MIT ([`LICENSE-CODE.md`](LICENSE-CODE.md)); data and biomechanics
   documentation are CC BY-NC-SA 4.0 ([`LICENSE-DATA.md`](LICENSE-DATA.md)). By
   contributing you agree your contributions are licensed the same way.

## Security & content issues

See [`SECURITY.md`](SECURITY.md) — email kyle@drivelinebaseball.com.
