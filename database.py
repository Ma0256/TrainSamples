# connect to database and export excel/CSV
import json
# need package mysql -connector
#import mysql.connector
#import pyodbc
import sqlalchemy as sa
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# get project dir
from TrainSamples import sample_dir


# return URL for local file data base
def get_acramos2VD_url():
    # MS access connector for "Sampledateien"
    access_db = rf"{sample_dir}\acramos2VD.mdb"
    connection_string = (
        r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
        #r"DBQ=C:\Users\Public\test\sqlalchemy-access\sqlalchemy_test.accdb;"
        rf"DBQ={access_db};"
        r"ExtendedAnsiSQL=1;"
    )
    url = sa.engine.URL.create(
        "access+pyodbc",
        query={"odbc_connect": connection_string}
    )
    return url


# connect to acramos MariaDB10 on NAS
def get_acrDb_url():
    # SQL access
    # Credentials were set up as a dictionary.
    credentials = '.acrDbCred'
    with open(Path.home() / credentials) as f:
        credentials = json.load(f)

        # Connect to the DB
        args = dict(
            user=credentials.get('username'),
            password=credentials.get('password'),
            database='acrDb',
            host='ADSIM.cs.technikum-wien.at',#'10.0.0.1',
            port=3307,
        )
        connection_url = f'mysql+mysqlconnector://{args["user"]}:{args["password"]}@{args["host"]}:{args["port"]}/{args["database"]}'
        return connection_url


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    to_excel = False

    # open MS access database "Sampledateien"
    #connection_url = get_acrDb_url()

    # create mariadb (it is a mysql fork) database server in terminal and read SQL dump file:
    # mariadb -u root -p -e"CREATE DATABASE acrdb"
    # mariadb acrdb -u root -p < acrDb_15-05-23.sql
    # connect to server via mariadb connector (from pip)
    connection_url = 'mariadb+mariadbconnector://root:adsim@localhost/acrDb'
    # connect to server via mysql connector
    #connection_url = 'mariadb+mysqlconnector://root:adsim@localhost/acrDb'
    #
    #connection_url = get_acramos2VD_url()

    engine = sa.create_engine(connection_url)
    metadata = sa.MetaData(bind=engine)
    #sa.MetaData.reflect(metadata)
    #some_table = sa.Table("tabmess", metadata, autoload_with=engine)
    #cols = some_table.c.keys()
    insp = sa.inspect(engine)
    tabs = insp.get_table_names()
    tabs = {k: (list(engine.execute(f'SELECT COUNT(*) FROM {k};'))[0][0], len(insp.get_columns(k))) for k in tabs}
    tabs = pd.DataFrame.from_dict(tabs, orient='index')
    print(tabs)
    table_name = 'tabzug'
    # iterate over index
    #df = pd.read_excel("tabzug.xlsx")
    for table_name in set(tabs.T) - {'tabzugax'}:
        rows = tabs.loc[table_name, 0]
        if rows:
            print(f"Reading SQL table {table_name} ...")
            df = pd.read_sql_table(table_name, engine)
            print(f"... to dataframe shape {df.shape}")
            if to_excel:
                # EXCEL is slower in writing and much slower in reading
                print(f"Writing Excel file {table_name} in {Path.cwd()} ...")
                df.to_excel(f"{table_name}.xlsx", index=False)
            else:
                print(f"Writing CSV file {table_name} in {Path.cwd()} ...")
                df.to_csv(f"{table_name}.csv", index=False,
                          sep=';', decimal=',')
            print(df)

