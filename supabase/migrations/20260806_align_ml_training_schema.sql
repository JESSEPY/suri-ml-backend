-- Aligns ml_soil_training_data with the real-data retraining schema
-- (docs/superpowers/specs/2026-08-05-real-soil-data-retraining-design.md).
-- The synthetic generator is fully retired; every row ingested from here on
-- is real, lab-sourced data via data/parse_real_soil_database.py.

-- potassium is now stored in its lab-native unit (cmol/kg), not converted.
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
  ADD COLUMN variety             TEXT,
  ADD COLUMN cultivation_method  TEXT,
  ADD COLUMN age_band            TEXT,
  ADD COLUMN target_unit         TEXT;

-- Backfill any existing rows before the NOT NULL constraints below apply.
-- target_unit follows data.nutrient_recommendation_guide.TARGET_UNIT exactly:
-- Coconut/Cacao/Mango/Banana are kg_tree, every other crop is kg_ha.
UPDATE public.ml_soil_training_data
SET target_unit = CASE
  WHEN crop IN ('Mango', 'Cacao', 'Coconut', 'Banana') THEN 'kg_tree'
  ELSE 'kg_ha'
END
WHERE target_unit IS NULL;

-- "None" is this schema's deliberate sentinel for "this axis doesn't apply
-- to this crop" (see ml/preprocessing.py's VARIETIES/CULTIVATION_METHODS/
-- AGE_BANDS and every resolve_* function in data/nutrient_recommendation_
-- guide.py) -- not a NULL, a real category the model trains on.
UPDATE public.ml_soil_training_data
SET variety            = COALESCE(variety, 'None'),
    cultivation_method = COALESCE(cultivation_method, 'None'),
    age_band           = COALESCE(age_band, 'None')
WHERE variety IS NULL OR cultivation_method IS NULL OR age_band IS NULL;

ALTER TABLE public.ml_soil_training_data
  ALTER COLUMN variety             SET NOT NULL,
  ALTER COLUMN variety             SET DEFAULT 'None',
  ALTER COLUMN cultivation_method  SET NOT NULL,
  ALTER COLUMN cultivation_method  SET DEFAULT 'None',
  ALTER COLUMN age_band            SET NOT NULL,
  ALTER COLUMN age_band            SET DEFAULT 'None',
  ALTER COLUMN target_unit         SET NOT NULL;

-- is_synthetic is dead weight now that the synthetic generator is retired --
-- every row is real. Drop it and its index.
-- CHECK BEFORE RUNNING: if this table already has rows, confirm none of them
-- are old synthetic data you still need to distinguish (SELECT is_synthetic,
-- count(*) FROM ml_soil_training_data GROUP BY is_synthetic) -- this step is
-- irreversible. Comment out this block and keep the column if in doubt.
DROP INDEX IF EXISTS idx_ml_training_synthetic;
ALTER TABLE public.ml_soil_training_data
  DROP COLUMN is_synthetic;
