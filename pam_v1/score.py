"""
Script to score a file of partner data for use in pre-approval

Before running, check variables:
    file_path: path of the whole file to be scored
    file_month: month of the file being scored (or other info eg. "mar_19_Worldpay_US")
        this will name the output file "mar_19_Worldpay_US_mailing_file_output.csv"
    trans_columns: this should match the column names in the file for the count of transactions
        oldest should be first
    TO_columns: column names of the £TO per month, again oldest should be first

"""
import pandas as pd
import numpy as np
from utilities import add_features, bin_model_features, buildset_woe_conversion, dwhQuery, ENGINE
from sklearn.linear_model import LogisticRegression
import json
from fast_to_SQL.fast_to_SQL import to_sql_fast
import time

MAILING_FILE = dwhQuery("""SELECT [com_id_h]
      ,[First_of_Decision_Month]
      ,[tenure_in_months]
      ,[MCC_DESC]
      ,[turnover_month_12]
      ,[turnover_month_11]
      ,[turnover_month_10]
      ,[turnover_month_9]
      ,[turnover_month_8]
      ,[turnover_month_7]
      ,[turnover_month_6]
      ,[turnover_month_5]
      ,[turnover_month_4]
      ,[turnover_month_3]
      ,[turnover_month_2]
      ,[turnover_month_1]
      ,[trans_num_month_12]
      ,[trans_num_month_11]
      ,[trans_num_month_10]
      ,[trans_num_month_9]
      ,[trans_num_month_8]
      ,[trans_num_month_7]
      ,[trans_num_month_6]
      ,[trans_num_month_5]
      ,[trans_num_month_4]
      ,[trans_num_month_3]
      ,[trans_num_month_2]
      ,[trans_num_month_1]
  FROM [pam].[dataset_PAMready]""")

trans_columns = [f"trans_num_month_{12 - i}" for i in range(12)]
TO_columns = [f"turnover_month_{12 - i}" for i in range(12)]


def score_file(trans_months, TO_months):
    global MAILING_FILE

    features = [
        "years_with_WP",
        "trend_Turnover_months_12_normed_1",
        "sum_trans_Q",
        "months_transacting_Y",
    ]

    bin_features = [
        "years_with_WP_bins",
        "trend_Turnover_months_12_normed_1_bins",
        "sum_trans_Q_bins",
        "months_transacting_Y_bins",
    ]

    MAILING_FILE = add_features(MAILING_FILE, trans_months[-3:], TO_months, TO_months)

    X = MAILING_FILE[features]
    X = bin_model_features(X)
    X = X[bin_features]
    for i in X.columns:
        X[i] = X[i].cat.codes


    with open("woe_classes.json", "r") as f:
        woe_classes = json.load(f)

    buildset_woe_conversion(
        X,
        woe_classes,
        f"WOE_binned.csv",
    )

    clf = LogisticRegression(solver="liblinear")

    train = pd.read_csv("all_buildset_WOE_binned_train.csv")

    clf.fit(train[bin_features], train.target)

    """
    tests to check to model_summary.xlsx, output should be:

    print(clf.coef_) = [[0.82591951 0.71898132 0.93585854 0.30410705]]
    print(clf.intercept_) = [-1.96478771]
    """

    woe_file = pd.read_csv(f"WOE_binned.csv")
    predictions = clf.predict_proba(woe_file)

    MAILING_FILE["score"] = predictions[:, 1]
    MAILING_FILE['Date_Ran'] = time.strftime('%Y-%m-%d %H:%M:%S')
    MAILING_FILE["trend_Turnover_months_12_normed_1"] = MAILING_FILE["trend_Turnover_months_12_normed_1"].apply(lambda x: x if np.isfinite([x]) else None)

    out_df = MAILING_FILE[["com_id_h"
     ,"First_of_Decision_Month"
     ,"tenure_in_months"
     ,"MCC_DESC"
     ,"years_with_WP"
     ,"trend_Turnover_months_12_normed_1"
     ,"sum_trans_Q"
     ,"months_transacting_Y"
     ,"score"
     ,"Date_Ran"]]
    to_sql_fast(out_df, 'pam.pamv1_output', ENGINE, if_exists='replace', custom={"[Date_Ran]" : "datetime"} )



print("\ncheck these months are in chronological order, oldest first:\n")
print(trans_columns)
print(TO_columns)

score_file(trans_columns, TO_columns)
