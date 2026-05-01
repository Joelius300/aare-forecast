CREATE USER reader WITH PASSWORD 'mängischlängtläse';

-- Allow connection to the database
GRANT CONNECT ON DATABASE aare_oraku_forecast TO reader;

-- Grant usage on public schema
GRANT USAGE ON SCHEMA public TO reader;

-- Grant read access to all existing tables
GRANT SELECT ON ALL TABLES IN SCHEMA public TO reader;

-- Grant read access to all future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT ON TABLES TO reader;

-- Grant usage on sequences (e.g. serial/identity columns)
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO reader;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT USAGE, SELECT ON SEQUENCES TO reader;
