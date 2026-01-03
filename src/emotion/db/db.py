import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import duckdb

from emotion.dataloader.popaneloader import POPANEDataLoader


class POPANEDB(POPANEDataLoader):
    __downloads_completed: bool = False
    __number_of_downloads: int = 0
    __duckdbpath: str = "data/processed/popane_emotion.db"
    __threadpool: ThreadPoolExecutor | None = None
    study1colums = "timestamp", "affect", "ECG", "EDA", "temp", "respiration", "SBP", "DBP", "marker"
    s1_dtype = ["DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE",
                "DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE", "INTEGER"]
    study2columns = "timestamp", "affect", "ECG", "EDA", "SBP", "DBP", "CO", "TPR", "marker"
    s2_dtype = ["DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE",
                "DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE", "INTEGER"]
    study3columns = "timestamp", "affect", "ECG", "EDA", "dzdt", "dz", "z0", "SBP", "DBP", "CO", "TPR", "marker"
    s3_dtype = ["DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE",
                "DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE", "INTEGER"]
    study4columns = "timestamp", "ECG", "EDA", "SBP", "DBP", "CO", "TPR", "marker"
    s4_dtype = ["DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE",
                "DOUBLE", "DOUBLE", "DOUBLE", "INTEGER"]
    study5columns = "timestamp", "affect", "ECG", "EDA", "SBP", "DBP", "CO", "TPR", "marker"
    s5_dtype = ["DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE",
                "DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE", "INTEGER"]
    study6columns = "timestamp", "affect", "ECG", "dzdt", "dz", "z0", "EDA", "SBP", "DBP", "CO", "TPR", "marker"
    s6_dtype = ["DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE",
                "DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE", "INTEGER"]
    study7columns = "timestamp", "affect", "ECG", "dzdt", "dz", "z0", "marker"
    s7_dtype = ["DOUBLE", "DOUBLE", "DOUBLE",
                "DOUBLE", "DOUBLE", "DOUBLE", "INTEGER"]

    study_columns_map = {
        1: (study1colums, s1_dtype),
        2: (study2columns, s2_dtype),
        3: (study3columns, s3_dtype),
        4: (study4columns, s4_dtype),
        5: (study5columns, s5_dtype),
        6: (study6columns, s6_dtype),
        7: (study7columns, s7_dtype),
    }

    def __init__(self, raw_dir: str = "data/raw/",
                 db_dir = "data/precessed/",
                 db_name: str = "popane_emotion.db") -> None:
        """
        raw_dir :
            where is raw data saved
        db_path :
            where is the db to be saved


        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(db_dir + 'db_creation.log', mode='w'),
                logging.StreamHandler()
            ]
        )
        super().__init__(raw_dir)

    def createDB(self):
        start_time = datetime.now()
        logging.info("Starting POPANE database creation")
        # create __duckdbpath directory if it doesn't exist
        db = duckdb.connect(database=self.__duckdbpath, read_only=False)
        db.execute(f"SET threads TO {os.cpu_count()}")

        try:
            import psutil
            available_gb = 8
            db.execute(f"SET memory_limit = '{available_gb}GB'")
            logging.info(
                f"Using {available_gb}GB memory, {os.cpu_count()} threads")
        except ImportError:
            db.execute("SET memory_limit = '8GB'")
            logging.info("Using 8GB memory (default)")

        popane_data_loader = POPANEDataLoader()
        total_files = 0

        for study_number in range(1, 8):
            study_meta = popane_data_loader.get_study_metadata(study_number)
            if study_meta is not None:
                if study_meta.FILE_PATH is None:
                    logging.warning(
                        f"Study {study_number} has no FILE_PATH in metadata, skipping.")
                    continue
                file_paths = list(study_meta.FILE_PATH)
                logging.info(f"Study {study_number}: {len(file_paths)} files")
                if file_paths:
                    db.execute(f"""
                               CREATE TABLE IF NOT EXISTS study{study_number} AS
                                SELECT
                                    CAST(NULL AS INTEGER) AS Subject_ID,
                                    *
                                FROM read_csv_auto(
                                    '{file_paths[0]}',
                                    skip=9,
                                    header=true,
                                    delim=',',
                                    sample_size=-1
                                )
                                WHERE 1=0;
                                """)
                    logging.info(f"Created study{study_number} table")

                # Batch insert all files
                for i, file_path in enumerate(file_paths):
                    self.__process_study_to_db(file_path, study_number, db)
                    total_files += 1
                    if (i + 1) % 10 == 0:
                        logging.info(
                            f"  Loaded {i + 1}/{len(file_paths)} files")

        elapsed = (datetime.now() - start_time).total_seconds()
        logging.info(
            f"Complete in {elapsed:.1f}s ({total_files} files, {total_files / elapsed:.1f} files/sec)")
        return db

    def create_table(self, db: duckdb.DuckDBPyConnection, table_name: str, columns: dict):
        columns_def = ", ".join(
            [f"{col} {dtype}" for col, dtype in columns.items()])
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_def});"
        db.execute(create_table_query)
        logging.info(f"Created table: {table_name}")

    def connect_db(self) -> duckdb.DuckDBPyConnection:
        db = duckdb.connect(database=self.__duckdbpath, read_only=False,
                            config={'threads': os.cpu_count() or 24, 'memory_limit': '8GB'})
        return db

    def _extract_subject_id(self, path: str) -> int:
        with open(path, "r") as f:
            for line in f:
                if line.startswith("#Subject_ID"):
                    return int(line.strip().split(",")[1])
        raise ValueError(f"Subject_ID not found in {path}")

    def __process_study_to_db(self, studypath: str, study_number: int, db: duckdb.DuckDBPyConnection):
        try:
            subject_id = self._extract_subject_id(studypath)
            db.execute(f"""
                INSERT INTO study{study_number}
                SELECT
                    {subject_id} AS Subject_ID,
                    *
                FROM read_csv_auto(
                    '{studypath}',
                    skip=9,
                    header=true,
                    delim=',',
                    sample_size=-1,
                    parallel=true,
                    ignore_errors=false
                );
            """)
        except Exception as e:
            file_name = os.path.basename(studypath)
            logging.error(f"Error loading {file_name}: {e}")


def main():
    """Test loading Study 1 Subject 6 data into DuckDB and export to Parquet"""

    createDB = POPANEDB()
    db = createDB.connect_db()
    createDB.create_table(db, "study1", dict(
        zip(createDB.study1colums, createDB.s1_dtype)))
    study1 = "study1"
    db.execute(f"""
        INSERT INTO {study1}
        SELECT * FROM read_csv('data/raw/study1/1_Baseline.csv',
            skip=9,
            header=true,
            delim=',',
            sample_size=-1);
    """)
    result = db.execute("SELECT * FROM study1 LIMIT 5").fetchall()
    print(result)
    # Show table info
    count = db.execute("SELECT COUNT(*) FROM study1").fetchone()
    print(f"\nTotal rows loaded: {count}")

    res = db.execute(f"""
       SELECT COLUMN_NAME
       FROM INFORMATION_SCHEMA.COLUMNS
       WHERE TABLE_NAME = 'study1';
    """)
    print("\nTable Columns:")
    for row in res.fetchall():
        print(row[0])


if __name__ == "__main__":
    main()
