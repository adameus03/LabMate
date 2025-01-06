CREATE TABLE IF NOT EXISTS public.users (
	user_id serial4 NOT NULL,
	passwd_hash varchar NOT NULL,
	"role" int4 DEFAULT 0 NOT NULL,
	ip_addr varchar NULL,
	registration_date date NOT NULL,
	last_login_date date NULL,
	username varchar NOT NULL,
  first_name varchar NULL,
  last_name varchar NULL,
  bio varchar NULL,
	num_requests int4 DEFAULT 0 NOT NULL,
	karma int4 DEFAULT 0 NOT NULL,
	email varchar NOT NULL,
	email_verified bool DEFAULT false NOT NULL,
	email_verification_token_hash varchar NOT NULL,
	sesskey_hash varchar NULL,
	last_usr_chng_date date NULL,
	sesskey_salt varchar NULL,
	passwd_salt varchar NOT NULL,
	email_verification_token_salt varchar NOT NULL,
	CONSTRAINT users_pk PRIMARY KEY (user_id),
	CONSTRAINT users_unique UNIQUE (username),
	CONSTRAINT users_unique_1 UNIQUE (email)
);

-- CREATE OR REPLACE FUNCTION get_user(user_id int4 DEFAULT NULL::int4, username varchar DEFAULT NULL::character varying, email varchar DEFAULT NULL::character varying)
-- 	RETURNS public.users AS 
-- $$

CREATE TABLE IF NOT EXISTS public.reagent_types(
	reagent_type_id serial4 NOT NULL,
	name varchar NOT NULL,
	CONSTRAINT reagent_types_pk PRIMARY KEY (reagent_type_id)
);

CREATE TABLE IF NOT EXISTS public.reagents(
	reagent_id serial4 NOT NULL,
	name varchar NOT NULL,
	vendor varchar NULL,
	reagent_type_id int4 NOT NULL,
	CONSTRAINT reagents_pk PRIMARY KEY (reagent_id),
	CONSTRAINT reagent_type_id_fk FOREIGN KEY (reagent_type_id) REFERENCES public.reagent_types(reagent_type_id)
);

CREATE TABLE IF NOT EXISTS public.faculties(
	faculty_id serial4 NOT NULL,
	name varchar NOT NULL,
	email_domain varchar NOT NULL,
	CONSTRAINT faculties_pk PRIMARY KEY (faculty_id)
);

CREATE TABLE IF NOT EXISTS public.labs(
	lab_id serial4 NOT NULL,
	name varchar NOT NULL,
	bearer_token_hash varchar NOT NULL,
	bearer_token_salt varchar NOT NULL,
	faculty_id int4 NOT NULL,
	CONSTRAINT labs_pk PRIMARY KEY (lab_id),
	CONSTRAINT faculty_id_fk FOREIGN KEY (faculty_id) REFERENCES public.faculties(faculty_id)
);

CREATE TABLE IF NOT EXISTS public.inventory(
	inventory_id serial4 NOT NULL,
	reagent_id int4 NOT NULL,
	date_added date NOT NULL,
	date_expire date NULL,
	lab_id int4 NOT NULL,
	epc TEXT NOT NULL,
	CONSTRAINT inventory_pk PRIMARY KEY (inventory_id),
	CONSTRAINT inventory_unique UNIQUE (epc),
	CONSTRAINT reagent_id_fk FOREIGN KEY (reagent_id) REFERENCES public.reagents(reagent_id),
	CONSTRAINT lab_id_fk FOREIGN KEY (lab_id) REFERENCES public.labs(lab_id)
);

-- Antennas
CREATE TABLE IF NOT EXISTS public.antennas(
	antenna_id serial4 NOT NULL,
	name varchar NULL,
	info varchar NULL,
	k int4 NOT NULL,
	lab_id int4 NOT NULL,
	CONSTRAINT antennas_pk PRIMARY KEY (antenna_id),
	CONSTRAINT lab_id_fk FOREIGN KEY (lab_id) REFERENCES public.labs(lab_id)
);

-- Inventory measurements
CREATE TABLE IF NOT EXISTS public.invm(
	"time" TIMESTAMPTZ NOT NULL,
	inventory_epc TEXT NOT NULL,
	antenna_id int4 NOT NULL,
	rx_signal_strength int4 NULL,
	read_rate int4 NULL,
	tx_power int4 NULL,
	read_latency int4 NULL,
	measurement_type int4 NULL,
	rotator_ktheta int4 NULL,
	rotator_kphi int4 NULL,
	FOREIGN KEY (inventory_epc) REFERENCES public.inventory(epc),
	FOREIGN KEY (antenna_id) REFERENCES public.antennas(antenna_id)
);

-- Make the TimescaleDB extension available
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- -- Commit changes, so that we can create the hypertable
-- COMMIT;

--Check if the TimescaleDB extension is available
DO $$
BEGIN
  -- Check if TimescaleDB is available
  IF NOT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb') THEN
    RAISE EXCEPTION 'TimescaleDB extension is not available!';
	ELSE
		RAISE NOTICE 'TimescaleDB extension is available';
  END IF;
END;
$$;

-- Make sure the create_hypertable function is available
do $$
begin
	if not exists (
        select * from pg_proc where proname = 'create_hypertable'
    ) then
        raise exception 'WHERE IS create_hypertable ???';
    else
        raise notice 'we have create_hypertable';
    END IF;
end;
$$;

-- Make sure we can access namespace _timescaledb_functions
do $$
begin
	if not exists (
		SELECT oid FROM pg_namespace WHERE nspname = '_timescaledb_functions'
	) then
		raise exception 'WHERE IS _timescaledb_functions ???';
	else
		raise notice 'we have _timescaledb_functions';
	END IF;
end;
$$;

-- Create a hypertable for invm (time-series data with TimescaleDB)
DO $$
begin
  -- Check if the invm table is already a hypertable
  IF NOT EXISTS (
    SELECT 1 
    FROM timescaledb_information.hypertables 
  	WHERE hypertable_name = 'invm'
  ) THEN
    -- Create the hypertable when we are sure it wasn't already created
		RAISE NOTICE 'Creating hypertable invm';
    PERFORM create_hypertable('invm', 'time');
		RAISE NOTICE 'Hypertable invm created';
	ELSE
	 	RAISE NOTICE 'Hypertable invm already exists, not creating it again';
  END IF;
END;
$$;
