from concurrent.futures import ThreadPoolExecutor
import threading
import os
from numpy import insert
import requests
import time

# from emotion.dataloader.popaneloader import POPANEDataLoader
# from emotion.studys.study import Study
import duckdb

from emotion.dataloader import POPANEDataLoader


class PROPANEdb(POPANEDataLoader):
    __downloads_completed: bool = False
    __number_of_downloads: int = 0
    __duckdbpath: str = "data/processed/propane_emotion.db"
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

    def __init__(self,
                 data_dir: str | None = "data/raw/"):
        super().__init__(data_dir)

    def createDB(self):
        db = duckdb.connect(database=self.__duckdbpath, read_only=False)
        threads = os.cpu_count()

        for study_number in range(1, 8):
            measurements, dtypes = self.study_columns_map[study_number]
            table_name = f"study{study_number}"
            columns = dict(zip(measurements, dtypes))
            self.create_table(db, table_name, columns)

        self.__threadpool = ThreadPoolExecutor(max_workers=threads)
        popane_data_loader = POPANEDataLoader()
        study_files = []
        for study_number in range(1, 8):
            study_meta = popane_data_loader.get_study_metadata(study_number)
            if study_meta is not None:
                for file_path in study_meta['file_path'].tolist():
                    study_files.append(
                        [{"file_path": file_path, "study_number": study_number}])

        with self.__threadpool as executor:
            futures = []
            while not study_files == []:
                file_study = study_files.pop()
                futures.append(executor.submit(
                    self.__process_study_to_db, file_study[0], file_study[0][1], db))
            for future in futures:
                future.result()

    def create_table(self, db: duckdb.DuckDBPyConnection, table_name: str, columns: dict):
        columns_def = ", ".join(
            [f"{col} {dtype}" for col, dtype in columns.items()])
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_def});"
        db.execute(create_table_query)

    def connect_db(self) -> duckdb.DuckDBPyConnection:
        db = duckdb.connect(database=self.__duckdbpath, read_only=False)
        return db

    def __process_study_to_db(self, studypath: str, study_number: int, db: duckdb.DuckDBPyConnection):
        insert_query = f"""INSERT INTO study{study_number} 
                   SELECT * FROM read_csv_auto('{studypath[0]}');
                   """
        db.execute(insert_query)


def main():
    """Test loading Study 1 Subject 6 data into DuckDB and export to Parquet"""

    createDB = PROPANEdb()
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
