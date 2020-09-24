import pandas as pd
import numpy as np
import sqlalchemy
from urllib.parse import quote_plus

params = quote_plus(
    """DRIVER={ODBC Driver 13 for SQL Server};SERVER=server-dw01.database.windows.net;DATABASE=DataWarehouseStaging;UID=dw_userlogin;PWD=kljDueK!7"""
)

ENGINE = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)


def cont_lookup(numChars, split_list, woe_list):
    # a lookup for a continuous variable into buckets with assigned values
    # all inputs need to be lists of the requisite size
    if len(split_list) + 1 == len(woe_list):
        pass
    else:
        print(numChars)
        raise ValueError 

    # make the lower bound list and the upper bound list
    low = np.array([-10**20] + split_list)
    high = np.array(split_list + [10**20])

    # tile the input array
    matrix = np.tile(numChars, (len(woe_list), 1))

    # ensure woe_list are in an array
    woe_list = np.array(woe_list)

    return np.dot((np.transpose(matrix) > low) &
                  (np.transpose(matrix) <= high), woe_list)

# cont_lookup([3,5,8,11,-90],[-10,1,5,9],[1,11,111,1111,11111])
# expected output is [111,111,1111,11111,1]


def cat_lookup(catChars, split_list, woe_list):
    # a lookup for a categorical variable with each category assigned a value

    # split_list expected to be list of lists,
    # this flattens into each unique element
    s = [x for t in split_list for x in t]

    # woe_list is the woe value to return for each list,
    # this flattens into the same length as s
    w = [woe_list[split_list.index(t)] for t in split_list for x in t]

    # prep the chars and output list to format for linear algebra
    matrix = np.tile(catChars, (len(w), 1))
    w = np.array(w)

    return np.dot(pd.Series(s) == pd.DataFrame(np.transpose(matrix)), w)

# cat_lookup(["B","D","E","C"],[["A","B","C"],["D","E"]],[1,11])
# expected output is [1,11,11,1]


def trendline(data, order=1):
    """
    Calculate slope of line of best fit in datarange  
    """
    coeffs = np.polyfit(range(len(data)), list(data), order)
    slope = coeffs[-2]
    return float(slope)


def bin_feature(df, field, bins, null_value, inf_value=False):
    copy_df = df.copy()

    copy_df[field] = copy_df[field].fillna(null_value)

    if inf_value:
        copy_df[field] = copy_df[field].replace(np.inf, inf_value)

    df[f"{field}_bins"] = pd.cut(copy_df[field], bins, right=False)

    return df


def add_12m_TO_trend(df, fields):
    output = []
    for i in df.index:
        if i % 1000 == 0:
            print(f"done {i}!")
        a = df.loc[i, fields]
        if sum(pd.notnull(a)) > 6:
            output.append(
                trendline(a[pd.notnull(a)]) / a[pd.notnull(a)].mean()
            )
        else:
            output.append(np.nan)
    
    return output


def year_tenure(df):
    return df["tenure_in_months"] // 12


def transaction_sum(df, columns):
    return df[columns].sum(axis=1)


def count_transacting_months(df, columns):
    return np.sum(pd.notnull(df[columns]), axis=1)


def add_features(df, trans_columns, count_columns, trend_columns):
    df = df.copy()

    df["years_with_WP"] = year_tenure(df)
    df["trend_Turnover_months_12_normed_1"] = add_12m_TO_trend(df, trend_columns)
    df["sum_trans_Q"] = transaction_sum(df, trans_columns)
    df["months_transacting_Y"] = count_transacting_months(df, count_columns) 

    return df


def bin_model_features(df):

    df = df.copy()

    df = bin_feature(df, "years_with_WP", [-1, 2, 5, np.inf], -1) # 3
    df = bin_feature(df, "months_transacting_Y", [0, 12, np.inf], 0) # 2
    df = bin_feature(df, "sum_trans_Q", [0, 100, 2000, np.inf], 0) # 3
    df = bin_feature(df, "trend_Turnover_months_12_normed_1", [-1, -0.5, 0, 0.03, np.inf], -1) # 4

    return df


def buildset_woe_conversion(df, woe_classes, outpath):
    """
    Takes a buildset and converts it to pre-calculated WOE bins, included in 
    woe_classes, in this format:

        "woe_classes": [
            {
                "var_name": "TO_month_1",
                "split_list": [2945.93, 5638.99],
                "woe_list": [0.49, -0.14],
                "is_discrete": 0,  
            },
            {...},
        ]
    """

    woe_cont_list = []
    woe_disc_list = []

    for i in woe_classes:
        if i["is_discrete"] == 0:
            woe_cont_list.append([i["var_name"], i["split_list"], i["woe_list"]])
        else:
            woe_disc_list.append([i["var_name"], i["split_list"], i["woe_list"]])

    for i in woe_cont_list:
        if len(i[1]) == len(i[2]):
            i[1] = i[1][:-1]
        else:
            pass

    woe_Df = df.copy()

    for i in woe_cont_list:
        if i[0] in woe_Df.columns:
            woe_Df[i[0]] = cont_lookup(woe_Df[i[0]], i[1], i[2])

    for i in woe_disc_list:
        if i[0] in woe_Df.columns:
            woe_Df[i[0]] = cat_lookup(woe_Df[i[0]], i[1], i[2])

    woe_Df.to_csv(outpath, index=False)
    print(f"Buildset converted and written to {outpath}")


def dwhQuery(sqlQuery):
    """
    Wrapper for turning SQL query to DWH
    into a pandas dataframe
    :param sqlQuery: SQL query to be processed
    :return: SQL query as a pandas DataFrame
    """
    return pd.read_sql(sql=sqlQuery, con=ENGINE)
