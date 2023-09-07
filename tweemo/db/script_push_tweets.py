from tweemo.db import IOSQL
import pandas as pd

if __name__ == "__main__":
    dbname = "tweets_db"
    iosql = IOSQL(dbname=dbname)
    df_for_interface = pd.read_csv(
        '/home/godwin/Documents/Uparis/M1MLSD2223/tweets-modeling/tweemo/tweets/data'
        '/tweets_for_clustering.csv')
    tb_name = "tweets_cluster_tb"
    iosql.write_table(tb_name=tb_name, df=df_for_interface)
