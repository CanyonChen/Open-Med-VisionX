# OpenMedVisionX Security Policy

## Supported versions

This project is currently pre-1.0. Security fixes are applied to the latest
source on the default branch. Older snapshots and locally modified builds are
not supported.

## Private reporting

Do not open a public issue for:

- a vulnerability that enables code execution, path traversal, data exposure,
  credential disclosure, or denial of service;
- an API key, private key, access token, or password found in source or history;
- DICOM, NIfTI, pixel data, metadata, or any other possible patient data found
  in the repository or an artifact;
- a malicious or unexpectedly privileged plugin.

Use GitHub's private vulnerability reporting for the repository. If it is not
available, contact a maintainer through a private channel shown on the
repository profile. Do not attach the sensitive file. Provide only the minimum
synthetic reproducer and a description of where the maintainer can verify the
issue privately.

A useful report includes the affected version or commit, platform, impact,
reproduction steps using synthetic data, and any proposed mitigation.

## Immediate response to exposed data

If a secret is exposed:

1. Revoke or rotate it immediately.
2. Stop using affected credentials and inspect provider audit logs.
3. Remove it from the working tree.
4. Rewrite Git history and purge published artifacts or caches before making
   the repository public again.
5. Add or strengthen an automated detection rule.

If possible medical or patient data is exposed:

1. Stop distribution and restrict access.
2. Do not copy, inspect, or redistribute more data than needed to identify the
   affected object.
3. Remove it from the working tree, Git history, releases, and mirrors under
   project control.
4. Notify the responsible data owner and follow applicable institutional and
   legal incident procedures.
5. Replace any test dependency with a runtime-generated synthetic fixture.

Deleting a file in a later commit does not remove it from Git history.

## Security boundaries

- A Python model plugin is arbitrary code. A separate process and Conda
  environment improve fault isolation but are not a security sandbox.
- Model files, plugins, and weights are untrusted until the user reviews their
  source and licenses.
- Cloud image transmission is disabled by default and requires revocable
  per-provider consent.
- Only a user-selected, previewed, metadata-stripped rendered slice may be sent
  to a provider. Original DICOM/NIfTI, full series, and DICOM metadata are
  prohibited.
- Burned-in text can remain identifying even after metadata removal.
- The application is not a medical device and is not suitable for clinical
  decisions.

## Repository policy

Run `python scripts/check_repository.py` before every commit. The check rejects
medical formats and signatures, archives, model artifacts,
executables, likely secrets, forbidden build directories, and oversized files.
Security checks reduce risk but do not prove that a file is anonymized or safe.
