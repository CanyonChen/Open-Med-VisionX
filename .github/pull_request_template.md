## Outcome

Describe the user-visible or architectural outcome and link the issue it
addresses.

## Scope

- Public interfaces changed:
- Optional dependencies changed:
- Security, privacy, or license impact:

## Verification

List the exact checks you ran and their results.

```text
python scripts/check_repository.py
ruff check src tests scripts
python -m pytest
```

## Checklist

- [ ] The change follows the Domain/IO/Algorithm/Inference/LLM/Runtime/Services/UI boundaries.
- [ ] Slow work supports progress and cancellation without blocking the GUI.
- [ ] Tests generate synthetic fixtures at runtime.
- [ ] I did not add medical images, patient data, archives, model weights, executables, build output, or credentials.
- [ ] Model/plugin code and weight licenses are documented when applicable.
- [ ] Cloud tests use a local mock and do not call a real provider.
- [ ] User-visible behavior and public interfaces are documented in both languages.
- [ ] `CHANGELOG.md` is updated when the change is notable.
- [ ] The repository policy check passes.
