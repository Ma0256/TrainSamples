import json
#import mysql.connector
#import pyodbc
import sqlalchemy as sa
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_acramos2VD_url():
    # MS access connector for "Sampledateien"
    access_root = rf"{str(Path.home())}\prj\Sampledateien\acramos2VD.mdb"
    connection_string = (
        r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
        #r"DBQ=C:\Users\Public\test\sqlalchemy-access\sqlalchemy_test.accdb;"
        rf"DBQ={access_root};"
        r"ExtendedAnsiSQL=1;"
    )
    url = sa.engine.URL.create(
        "access+pyodbc",
        query={"odbc_connect": connection_string}
    )
    return url


# connect to acramos MariaDB10
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
    # open MS access database "Sampledateien"
    #connection_url = get_acrDb_url()
    connection_url = get_acramos2VD_url()
    engine = sa.create_engine(connection_url)
    print(engine.table_names())
    table_name = 'TabZug'
    df = pd.read_sql_table(table_name, engine)
    df.to_excel(f"acrDb_{table_name}.xlsx")
    print(df)
