"""Session-wide test bootstrap.

main.py loads whatever model artifact is on disk at *import* time
(boot_version(), executed as a module-level statement), and most of this
suite exercises /predict, /model/versions, /model/confidence, and friends
against whatever got booted. The repo intentionally ships with
ml/artifacts/ empty (see the "reset to a clean slate" chore commit -- the
real demo trains its first model live from the admin UI), so a fresh
checkout -- a new clone, or CI -- has nothing to boot, and every test that
assumes a live model 503s instead of exercising the thing it's testing.

pytest_configure runs before any test module is collected/imported, which
means before main.py has been imported anywhere -- the last point at which
a bootstrap artifact can be written before boot_version() reads the (now
non-empty) directory. Training is fully synthetic (tests.fixtures'
sample_training_frame), so this needs no network access and no real
Supabase credentials, matching CI's dummy SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY.

Skipped entirely if a real artifact is already present (e.g. a developer
running the suite locally, mid-session, after their own retrain) -- and in
that case nothing is cleaned up afterward, since nothing was created.
Otherwise the bootstrap artifact is removed again in pytest_unconfigure, so
a local `pytest` run doesn't silently leave the clean-slate demo state
dirty for whoever's working here interactively.

Also populates data.nutrient_recommendation_guide's module state (GUIDE_ROWS,
CROPS, VARIETIES_BY_CROP, etc.) here, before anything else imports that
module. Since the crop-nutrient-reference-cms work
(docs/superpowers/plans/2026-09-01-crop-nutrient-reference-cms.md Task 4),
that state is no longer a hardcoded literal -- it's loaded via
reload_guide(), which normally hits a live Supabase project. Tests inject a
fake client instead (tests/guide_fixture.py, built from the committed
tests/fixtures/crop_reference_seed.csv, the same historical dataset the old
literal held), so the whole suite -- including this file's own
sample_training_frame() call below, which needs guide.CROPS non-empty --
stays network-free.
"""

import os
import shutil

_bootstrapped_version: str | None = None


def pytest_configure(config):
    global _bootstrapped_version

    import data.nutrient_recommendation_guide as guide
    from tests.guide_fixture import load_fixture_client

    guide.reload_guide(client=load_fixture_client())

    from ml.model_state import available_versions
    from ml.train import ARTIFACT_ROOT, train
    from tests.fixtures import sample_training_frame

    if available_versions(ARTIFACT_ROOT):
        return

    metadata = train(sample_training_frame(num_records=200, seed=42))
    _bootstrapped_version = metadata["version"]


def pytest_unconfigure(config):
    if _bootstrapped_version is None:
        return

    from ml.train import ARTIFACT_ROOT

    shutil.rmtree(os.path.join(ARTIFACT_ROOT, _bootstrapped_version), ignore_errors=True)
