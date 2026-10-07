-- Executed by the postgres image on first initialisation of an empty data volume.
-- The main database (judge_check) is created by POSTGRES_DB; here we add a
-- separate database that the test suite migrates and wipes freely (D-009).
CREATE DATABASE judge_check_test OWNER judge_check;
