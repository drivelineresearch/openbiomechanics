## Summary

<!-- What changed, and what user/contributor problem does it solve? -->

## Public impact

<!-- Note API, dataset, documentation, download, compatibility, or scientific-interpretation effects. Write "None" where appropriate. -->

## Verification

<!-- List exact commands and outcomes. Include fresh-clone evidence for onboarding/example changes. -->

## Checklist

- [ ] The change is focused and does not include unrelated local files.
- [ ] I ran the relevant checks from `CONTRIBUTING.md` and reported their results.
- [ ] Examples and paths work from a fresh clone without relying on ignored local data.
- [ ] CSV column names, join keys, units, and sampling rates were checked against authoritative files/docs.
- [ ] Generated dictionaries are current (`python3 scripts/build_data_dictionary.py --check`).
- [ ] No large C3D/full-signal/media/model/installer artifact is committed.
- [ ] Code changes use `LICENSE-CODE.md`; data and biomechanics-doc changes preserve `LICENSE-DATA.md`.
- [ ] Any new limitation, migration step, or follow-up is documented.
