-- SURI Fertilizer ML Pipeline tables.
-- Training corpus is keyed by (laboratory_code, data_source): each soil record
-- contributes three rows, one per data source view.

CREATE TABLE public.ml_soil_training_data (
  laboratory_code      TEXT NOT NULL,
  data_source          TEXT NOT NULL CHECK (data_source IN ('lab', 'stk', 'iot')),
  sampling_site        TEXT,
  date_of_sampling     DATE,
  crop                 TEXT NOT NULL,
  farm_area_sqm        NUMERIC,
  ph                   NUMERIC,
  organic_matter_pct   NUMERIC,
  total_nitrogen_pct   NUMERIC,
  phosphorus_ppm       NUMERIC,
  potassium_ppm        NUMERIC,
  temp                 NUMERIC,
  moist                NUMERIC,
  ec                   NUMERIC,
  sal                  NUMERIC,
  ph_remark            TEXT,
  om_remark            TEXT,
  p_remark             TEXT,
  k_remark             TEXT,
  target_n_kg_ha       NUMERIC NOT NULL,
  target_p2o5_kg_ha    NUMERIC NOT NULL,
  target_k2o_kg_ha     NUMERIC NOT NULL,
  app1_formulation     TEXT NOT NULL,
  app2_formulation     TEXT NOT NULL,
  is_synthetic         BOOLEAN NOT NULL DEFAULT TRUE,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (laboratory_code, data_source)
);

CREATE INDEX idx_ml_training_crop ON public.ml_soil_training_data(crop);
CREATE INDEX idx_ml_training_source ON public.ml_soil_training_data(data_source);
CREATE INDEX idx_ml_training_synthetic ON public.ml_soil_training_data(is_synthetic);

-- Append-only audit log of live predictions, for retraining and traceability.
CREATE TABLE public.ml_fertilizer_predictions (
  id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  laboratory_code      TEXT,
  data_source          TEXT NOT NULL,
  request_payload      JSONB NOT NULL,
  response_payload     JSONB NOT NULL,
  model_version        TEXT NOT NULL,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ml_predictions_created ON public.ml_fertilizer_predictions(created_at DESC);
CREATE INDEX idx_ml_predictions_version ON public.ml_fertilizer_predictions(model_version);

ALTER TABLE public.ml_soil_training_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ml_fertilizer_predictions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Admin access to ml_soil_training_data" ON public.ml_soil_training_data
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM public.profiles
      WHERE profiles.id = auth.uid()
      AND profiles.role IN ('admin', 'superadmin')
    )
  );

CREATE POLICY "Admin access to ml_fertilizer_predictions" ON public.ml_fertilizer_predictions
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM public.profiles
      WHERE profiles.id = auth.uid()
      AND profiles.role IN ('admin', 'superadmin')
    )
  );
