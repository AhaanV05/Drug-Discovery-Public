-- =========================
-- Drug Discovery Database Schema
-- Supabase PostgreSQL Setup
-- =========================

-- =========================
-- 0) Keep things tidy with a dedicated schema
-- =========================
CREATE SCHEMA IF NOT EXISTS dd;
GRANT USAGE ON SCHEMA dd TO public;

-- =========================
-- 1) Core table: compounds
-- =========================
CREATE TABLE IF NOT EXISTS dd.compounds (
  molecule_chembl_id TEXT PRIMARY KEY,
  canonical_smiles   TEXT NOT NULL,

  -- basic physchem (either from ChEMBL or computed later)
  mw   REAL,
  alogp REAL,
  hbd  INT,
  hba  INT,
  tpsa REAL,

  -- additional metadata
  max_phase INT,
  first_approval INT,
  pref_name TEXT,

  -- provenance / bookkeeping
  source TEXT NOT NULL DEFAULT 'chembl',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_compounds_smiles ON dd.compounds USING btree (canonical_smiles);
CREATE INDEX IF NOT EXISTS idx_compounds_mw ON dd.compounds USING btree (mw);
CREATE INDEX IF NOT EXISTS idx_compounds_alogp ON dd.compounds USING btree (alogp);
CREATE INDEX IF NOT EXISTS idx_compounds_hbd ON dd.compounds USING btree (hbd);
CREATE INDEX IF NOT EXISTS idx_compounds_hba ON dd.compounds USING btree (hba);
CREATE INDEX IF NOT EXISTS idx_compounds_tpsa ON dd.compounds USING btree (tpsa);

-- Ensure no duplicate SMILES if you want strict uniqueness (optional)
-- CREATE UNIQUE INDEX IF NOT EXISTS uq_compounds_smiles ON dd.compounds (canonical_smiles);

-- =========================
-- 2) Dataset versions (so you can regenerate/re-filter later)
-- =========================
CREATE TABLE IF NOT EXISTS dd.dataset_versions (
  dataset_version_id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  filters_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_dataset_versions_name ON dd.dataset_versions (name);

-- =========================
-- 3) Membership of compounds in a dataset version
--    (lets you keep multiple curated sets without duplicating compounds)
-- =========================
CREATE TABLE IF NOT EXISTS dd.dataset_compounds (
  dataset_version_id BIGINT NOT NULL REFERENCES dd.dataset_versions(dataset_version_id) ON DELETE CASCADE,
  molecule_chembl_id TEXT NOT NULL REFERENCES dd.compounds(molecule_chembl_id) ON DELETE CASCADE,
  split TEXT,          -- 'train' | 'val' | 'test' (optional)
  weight REAL,         -- optional sampling weight
  added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (dataset_version_id, molecule_chembl_id)
);

CREATE INDEX IF NOT EXISTS idx_dataset_compounds_split
  ON dd.dataset_compounds (dataset_version_id, split);

-- Composite index for fast reverse lookup of dataset membership
CREATE INDEX IF NOT EXISTS idx_dataset_compounds_molecule
  ON dd.dataset_compounds (molecule_chembl_id);

-- =========================
-- 4) Activities table (bioactivity data)
-- =========================
CREATE TABLE IF NOT EXISTS dd.activities (
  activity_id BIGINT PRIMARY KEY,
  molecule_chembl_id TEXT NOT NULL REFERENCES dd.compounds(molecule_chembl_id) ON DELETE CASCADE,
  assay_chembl_id TEXT,
  target_chembl_id TEXT,
  standard_type TEXT,
  standard_relation TEXT,
  standard_value REAL,
  standard_units TEXT,
  pchembl_value REAL,
  data_validity_comment TEXT,
  bao_endpoint TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_activities_molecule ON dd.activities (molecule_chembl_id);
CREATE INDEX IF NOT EXISTS idx_activities_assay ON dd.activities (assay_chembl_id);
CREATE INDEX IF NOT EXISTS idx_activities_target ON dd.activities (target_chembl_id);
CREATE INDEX IF NOT EXISTS idx_activities_type ON dd.activities (standard_type);
CREATE INDEX IF NOT EXISTS idx_activities_pchembl ON dd.activities (pchembl_value);

-- Composite index for fast activity lookups by molecule + type
CREATE INDEX IF NOT EXISTS idx_activities_mol_type
  ON dd.activities (molecule_chembl_id, standard_type);

-- =========================
-- 5) Assays table
-- =========================
CREATE TABLE IF NOT EXISTS dd.assays (
  assay_chembl_id TEXT PRIMARY KEY,
  assay_type TEXT,
  confidence_score INT,
  bao_format TEXT,
  bao_endpoint TEXT,
  target_chembl_id TEXT,
  description TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_assays_type ON dd.assays (assay_type);
CREATE INDEX IF NOT EXISTS idx_assays_target ON dd.assays (target_chembl_id);
CREATE INDEX IF NOT EXISTS idx_assays_confidence ON dd.assays (confidence_score);

-- =========================
-- 6) Targets table
-- =========================
CREATE TABLE IF NOT EXISTS dd.targets (
  target_chembl_id TEXT PRIMARY KEY,
  target_type TEXT,
  pref_name TEXT,
  organism TEXT,
  taxonomy_id INT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_targets_type ON dd.targets (target_type);
CREATE INDEX IF NOT EXISTS idx_targets_organism ON dd.targets (organism);

-- =========================
-- 7) Optional: RDKit-computed descriptors (store as key/value)
--    Use when you compute extra features for ML
-- =========================
CREATE TABLE IF NOT EXISTS dd.compound_features (
  molecule_chembl_id TEXT NOT NULL REFERENCES dd.compounds(molecule_chembl_id) ON DELETE CASCADE,
  feature_set TEXT NOT NULL, -- e.g. 'rdkit_basic_v1'
  features JSONB NOT NULL,   -- arbitrary descriptor dict
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (molecule_chembl_id, feature_set)
);

CREATE INDEX IF NOT EXISTS idx_compound_features_gin
  ON dd.compound_features USING gin (features);

-- =========================
-- 8) Optional: fingerprints for fast similarity later
--    (lightweight version: store as bytea)
--    If you later install RDKit or pgvector, we can upgrade this.
-- =========================
CREATE TABLE IF NOT EXISTS dd.compound_fingerprints (
  molecule_chembl_id TEXT NOT NULL REFERENCES dd.compounds(molecule_chembl_id) ON DELETE CASCADE,
  fp_type TEXT NOT NULL,        -- 'morgan_2048_r2'
  fp_bytes BYTEA NOT NULL,      -- packed bit vector
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (molecule_chembl_id, fp_type)
);

-- =========================
-- 9) Create read-only role (optional but recommended)
-- =========================
-- Check if role exists first
-- DO $$
-- BEGIN
--   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'chembl_readonly') THEN
--     CREATE ROLE chembl_readonly WITH LOGIN PASSWORD 'readonly_password_change_me';
--   END IF;
-- END
-- $$;

-- =========================
-- 10) Permissions for read-only role
-- =========================
GRANT USAGE ON SCHEMA dd TO chembl_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA dd TO chembl_readonly;

-- Ensure future tables in dd are readable too
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA dd
GRANT SELECT ON TABLES TO chembl_readonly;

-- =========================
-- Done! Schema is ready for data ingestion
-- =========================
