import os
import pandas as pd
import psycopg2  # need in some cases: > sudo apt-get install libpq-dev
from sqlalchemy import create_engine


class IOSQL:
    USERNAME = "postgres"
    PASSWORD = "mysecretpassword"
    HOST = "localhost"
    PORT = "5432"

    def __init__(
            self,
            dbname: str,
            user: str = USERNAME,
            password: str = PASSWORD,
            host: str = HOST,
            port: str = PORT
    ):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=password,
            host=self.host,
            port=self.port
        )

    def read_table(
            self,
            tb_name: str,
            columns=None
    ) -> pd.DataFrame:
        if columns is None:
            columns = ['*']
        query_rd = "SELECT {} FROM {}".format(",".join(columns), tb_name)
        cur = self.conn.cursor()
        try:
            cur.execute(query_rd)
        except psycopg2.Error as e:
            return e
        rows = cur.fetchall()
        try:
            cur.execute("SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = N{}"
                        .format("\'" + tb_name + "\'"))
        except psycopg2.Error as e:
            return e
        columns = [c[0] for c in cur.fetchall()] if "*" in columns else columns
        cur.close()
        return pd.DataFrame(rows, columns=columns)

    def write_table(
            self,
            tb_name: str,
            df: pd.DataFrame,
            default_dir=".",
            if_exist=False
    ) -> None:

        # build sqlalchemy format: postgresql+psycopg2://user:password@host:port/dbname
        connect = "postgresql+psycopg2://%s:%s@%s:5432/%s" % (
            self.user,
            self.password,
            self.host,
            self.dbname
        )
        engine = create_engine(connect)
        df.to_sql(tb_name, con=engine, index=False, if_exists="replace")

    def close(self):
        self.conn.close()
