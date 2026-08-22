-- Model version registry for the admin retraining pipeline
-- (docs/superpowers/specs/2026-08-18-training-data-pipeline-design.md §5).
--
-- Replaces main.py's previous implicit "whichever artifact directory name
-- sorts latest" rule with an explicit, auditable record of which version is
-- live, when it was trained, how it scored, and who promoted it.

CREATE TABLE public.ml_model_versions (
  version       TEXT PRIMARY KEY,
  trained_at    TIMESTAMPTZ NOT NULL,
  row_count     INTEGER NOT NULL,
  metrics       JSONB NOT NULL,
  promoted_at   TIMESTAMPTZ,
  promoted_by   TEXT,
  is_current    BOOLEAN NOT NULL DEFAULT FALSE,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- At most one live version at a time. A partial unique index (rather than a
-- CHECK) is what makes "exactly one true" enforceable at the database level.
CREATE UNIQUE INDEX idx_ml_model_versions_single_current
  ON public.ml_model_versions(is_current)
  WHERE is_current;

CREATE INDEX idx_ml_model_versions_trained_at
  ON public.ml_model_versions(trained_at DESC);

ALTER TABLE public.ml_model_versions ENABLE ROW LEVEL SECURITY;

-- Same admin-only policy shape the sibling ML tables already use
-- (supabase/migrations/20260803_ml_fertilizer_tables.sql). The service role
-- bypasses RLS entirely, which is how the ML backend itself reads and writes
-- this table; this policy is for direct dashboard/SQL access by humans.
CREATE POLICY "Admin access to ml_model_versions" ON public.ml_model_versions
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM public.profiles
      WHERE profiles.id = auth.uid()
        AND profiles.role IN ('admin', 'superadmin')
    )
  );

-- Private bucket holding a copy of every trained artifact directory, so a
-- rebuilt EC2 instance can recover any version rather than only the one
-- pinned into git (see .gitignore's ml/artifacts/* exception).
INSERT INTO storage.buckets (id, name, public)
VALUES ('ml-model-artifacts', 'ml-model-artifacts', FALSE)
ON CONFLICT (id) DO NOTHING;
