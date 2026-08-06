-- Aligns ml_soil_training_data with the real-data retraining schema
-- (docs/superpowers/specs/2026-08-05-real-soil-data-retraining-design.md).
-- The synthetic generator is fully retired; every row ingested from here on
-- is real, lab-sourced data via data/parse_real_soil_database.py.
--
-- This table (ml_soil_training_data only -- NOT ml_fertilizer_predictions,
-- the separate JSONB-based audit log, which needs no schema change and
-- whose historical records remain valid regardless) has never successfully
-- received real data (the first real ingestion attempt failed before this
-- migration existed -- see
-- docs/superpowers/specs/2026-08-06-real-soil-data-retraining-results.md).
-- Any row already in this table predates the real pipeline entirely, so
-- rather than attempt to convert old values into the new schema (the old
-- potassium_ppm numbers can't become correct potassium_cmol_kg values
-- without knowing which specific conversion, if any, they need), this
-- migration deletes every pre-existing row outright. There is no real data
-- here to lose.

DELETE FROM public.ml_soil_training_data;

-- potassium is now stored in its lab-native unit (cmol/kg), not converted.
-- Safe now that no pre-existing row survives to carry a stale ppm value
-- under the new column name.
ALTER TABLE public.ml_soil_training_data
  RENAME COLUMN potassium_ppm TO potassium_cmol_kg;

-- Target columns lose their _kg_ha suffix: unit now varies by crop
-- (kg/ha for field crops, kg/tree for Coconut/Cacao/Mango/Banana) and is
-- carried explicitly in the new target_unit column below.
ALTER TABLE public.ml_soil_training_data
  RENAME COLUMN target_n_kg_ha TO target_n;
ALTER TABLE public.ml_soil_training_data
  RENAME COLUMN target_p2o5_kg_ha TO target_p2o5;
ALTER TABLE public.ml_soil_training_data
  RENAME COLUMN target_k2o_kg_ha TO target_k2o;

ALTER TABLE public.ml_soil_training_data
  ADD COLUMN variety             TEXT NOT NULL DEFAULT 'None',
  ADD COLUMN cultivation_method  TEXT NOT NULL DEFAULT 'None',
  ADD COLUMN age_band            TEXT NOT NULL DEFAULT 'None',
  ADD COLUMN target_unit         TEXT NOT NULL DEFAULT 'kg_ha';

-- No pre-existing rows survive the DELETE above, so no backfill UPDATE is
-- needed for the four columns just added -- every future row is written by
-- data/ingest_to_supabase.py, which always supplies all four explicitly.

-- is_synthetic is dead weight now that the synthetic generator is retired
-- and every pre-existing row (synthetic or not) has just been deleted.
DROP INDEX IF EXISTS idx_ml_training_synthetic;
ALTER TABLE public.ml_soil_training_data
  DROP COLUMN is_synthetic;
