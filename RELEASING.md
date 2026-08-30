# Release Guide

Large C3Ds, full-signal tables, computer-vision media, and third-party installers belong in GitHub Releases rather than Git history. Release tags and the committed checksum manifest form part of the dataset's reproducibility record.

## Principles

- Publish a new immutable tag for changed data; do not replace an asset under an existing tag.
- Keep code/data licensing and provenance visible in the release notes.
- Verify filenames, byte sizes, and SHA-256 digests before updating the downloader.
- Do not publish private source material, credentials, direct athlete identifiers, or assets outside the approved public sample.

## Dataset release checklist

1. Produce the release assets in a controlled workspace and record their source snapshot and generation process.
2. Confirm archive contents and naming conventions. Dataset assets use `pitching_*.zip` and `hitting_*.zip`; raw archives are named `pitching_c3d.zip` and `hitting_c3d.zip`.
3. Compute local SHA-256 values and retain the receipt.
4. Create a new GitHub Release tag and upload the assets.
5. Compare GitHub's stored digests and sizes with the local receipt:

   ```bash
   gh api repos/drivelineresearch/openbiomechanics/releases/tags/TAG \
     --jq '.assets[] | [.name, (.size|tostring), .digest] | @tsv'
   ```

6. Update the tag in `scripts/download_data.sh`, all user-facing documentation, and [`scripts/release_checksums.sha256`](scripts/release_checksums.sha256).
7. Run repository quality checks and perform at least one download into a disposable fresh checkout. Confirm C3Ds extract, full-signal archives remain zipped, and no downloaded artifact appears in `git status`.
8. Add a dated [`CHANGELOG.md`](CHANGELOG.md) entry with scope, compatibility notes, and known limitations.
9. If a DOI is minted, add the concept/version DOI and release date to [`CITATION.cff`](CITATION.cff) and validate GitHub's rendered citation.

The `OBP_GITHUB_REPOSITORY` environment variable may point the downloader at a staging fork for release-candidate validation; the default always uses `drivelineresearch/openbiomechanics` so ordinary forks still download the canonical public assets.
