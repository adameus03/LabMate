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

