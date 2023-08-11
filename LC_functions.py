import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
import xgboost as xgb

def print_def(variable_name):
    """
    Pass in a variable name as a string, print its defintion.
    Definitions included with the original dataset. 
    Only works with ORIGINAL feature names.
    """
    
    try:
        if  not ("definitions_" in globals()):
            try:
                global definitions_
                definitions_ = pd.read_csv("LC_definitions.csv")       
            except:
                print(f"Could not find \"LC_definitions.csv\" in {pwd}.")
        print(variable_name, "-", definitions_.loc[definitions_.LoanStatNew == variable_name, "Description"].iloc[0])
    except:
        print(f"***{variable_name}: New feature, not in original dictionary.")


#------------------------------------------------
#------------------------------------------------


def success_rates(_df, _feature, _target):
    """
    Attributes
    ----------
    _df: pandas DataFrame.
    _feature: string, categorical feature name.
    _target: string, binary feature, usually target.
    
    Returns
    -------
    Dictionary with Feature: Success Rate ([0,1] range) pairs.
    """
    
    _sr_dict = {}
    for i in df[_feature].value_counts().index:
        _sr_dict[i] =(_df.loc[_df[_feature]==i, _target].sum()/(df.loc[df[_feature]==i, _target].shape[0])).round(2)
    
    return _sr_dict


#------------------------------------------------
#------------------------------------------------


def load_lending_club(path_):
    date_cols = ["issue_d", "earliest_cr_line", "last_pymnt_d", "last_credit_pull_d"]
    return pd.read_csv(path_, 
                 low_memory=False, 
                 parse_dates=date_cols, 
                 date_format="%b-%Y")


#------------------------------------------------
#------------------------------------------------


def tweak_lending_club(df_, ohe=False):
    drop_cols = ["debt_settlement_flag", "Unnamed: 0",  
                 "last_fico_range_low", "num_bc_sats", "num_bc_tl", 
                 "fico_range_low", "num_rev_accts", "out_prncp", 
                 "out_prncp_inv", "out_prncp_inv", "policy_code", 
                 "funded_amnt_inv", "loan_amnt", 
                 "collection_recovery_fee", "last_credit_pull_d", "id", 
                 "url", "pymnt_plan", "emp_title", 
                 "title", "zip_code", "verification_status", "last_pymnt_d"]
    high_nan_cols = ['all_util', 'annual_inc_joint', 'deferral_term', 
                     'dti_joint', 'hardship_amount', 'hardship_dpd', 'hardship_end_date', 
                     'hardship_last_payment_amount', 'hardship_length', 'hardship_loan_status', 
                     'hardship_payoff_balance_amount', 'hardship_reason', 'hardship_start_date', 
                     'hardship_status', 'hardship_type', 'il_util', 
                     'inq_fi', 'inq_last_12m', 'max_bal_bc', 
                     'mths_since_last_delinq', 'mths_since_last_major_derog', 'mths_since_last_record', 
                     'mths_since_rcnt_il', 'mths_since_recent_bc_dlq', 'mths_since_recent_revol_delinq', 
                     'next_pymnt_d', 'open_acc_6m', 'open_act_il', 
                     'open_il_12m', 'open_il_24m', 'open_rv_12m', 
                     'open_rv_24m', 'orig_projected_additional_accrued_interest', 'payment_plan_start_date', 
                     'revol_bal_joint', 'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med', 
                     'sec_app_earliest_cr_line', 'sec_app_fico_range_high', 'sec_app_fico_range_low', 
                     'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_num_rev_accts', 
                     'sec_app_open_acc', 'sec_app_open_act_il', 'sec_app_revol_util', 
                     'total_bal_il', 'total_cu_tl', 'verification_status_joint']
    transformed_cols = ["term", "initial_list_status", 
                        "application_type", "hardship_flag", "emp_length", 
                        "int_rate", "revol_util"]
    object_cols = ["grade", "sub_grade", "home_ownership", 
                   "purpose", "addr_state", "issue_date_month"]
    
    
    LC_df_ = df_.assign(target = np.where((df_.loan_status == "Charged Off") | 
                                        (df_.loan_status == "Does not meet the credit policy. Status:Charged Off") | 
                                        (df_.loan_status == "Default"), 
                                        1, 0), 
                      term_36months = np.where(df_.term == "60 months", 0, 1), 
                      initial_list_status_w = np.where(df_.initial_list_status == "f", 0, 1), 
                      individual = np.where(df_.application_type == "Joint App", 0, 1),
                      hardhship = np.where(df_.hardship_flag == "Y", 1, 0),
                      employed_over_10yrs = np.where(df_.emp_length == "10+ years", 1, 0),
                      emp_years = df_.emp_length.map({"< 1 year": 0, "1 year": 1, "2 years": 2, 
                                                        "3 years": 3, "4 years": 4, "5 years": 5, 
                                                        "6 years": 6, "7 years": 7, "8 years": 8, 
                                                        "9 years": 9, "10+ years": 10}),
                      interest_rate = df_.int_rate.str.replace('%', '').astype(float),
                      revolving_util = df_.revol_util.str.replace('%', '').astype(float),
                      issue_date_month = df_.issue_d.dt.month.astype(str),
                      issue_date_year = df_.issue_d.dt.year,
                      years_since_earliest_cr_line = df_.issue_d.dt.year - df_.earliest_cr_line.dt.year,
                     ).drop(labels=drop_cols+high_nan_cols+transformed_cols, axis=1)
    
    print("Note: \"target\" series generated from \"loan_status\" series.\"loan_status\" not dropped.")
        
    if ohe:
        return(pd.get_dummies(LC_df_, columns=object_cols).fillna(method="ffill"))
    else:
        return LC_df_


#------------------------------------------------
#------------------------------------------------


def lift_chart(model, X_trainf, X_testf, y_trainf, y_testf, bins=10):
    """
    This function takes a fitted sklearn.linear_model LogisticRegression object along
    with a split dataset and plt.show()'s  a lift chart. 
    Requires matplotlib.pyplot imported as plt.
    Requires numpy imported as np
    
    Parameters
    ----------
    model: sklearn.linear_model.LogisticRegression()
        A fitted LogisticRegression object
    X_trainf: pd.DataFrame.
        Feature training data
    X_testf: pd.DataFrame.
        Feature test/validation data
    y_trainf: pd.Series
        Target training data.
    y_testf: pd.Series
        Target test/validation data.
    bins: Non-zero natural number
        Number of bins to split the data into for plotting.
        Default is 10 bins.
    """

    preds_train = model.predict_proba(X_trainf)[:, 1]
    preds_test = model.predict_proba(X_testf)[:, 1]
    
    _df_train = pd.DataFrame()
    _df_train["target"] = y_trainf.copy()
    _df_train["preds"] = preds_train
    _df_train = _df_train.sort_values(by="preds", ascending=False)
    
    _df_test = pd.DataFrame()
    _df_test["target"] = y_testf.copy()
    _df_test["preds"] = preds_test
    _df_test = _df_test.sort_values(by="preds", ascending=False)
    
    max_bias_train = _df_train.target.mean()
    max_bias_test = _df_test.target.mean()
    bin_labels = list(np.arange(1, bins+1, 1))
    
    df_size = len(_df_train)
    bin_size = df_size//bins
    bin_list = [0 + bin_size*i for i in range(bins)]
    bin_probs_train = [_df_train["target"][bin_list[i]:bin_list[i+1]].mean()/max_bias_train for i in range(bins-1)]
    bin_probs_train.append(_df_train["target"][bin_list[bins-1]:].mean())
    
    df_size = len(_df_test)
    bin_size = df_size//bins
    bin_list = [0 + bin_size*i for i in range(bins)]
    bin_probs_test = [_df_test["target"][bin_list[i]:bin_list[i+1]].mean()/max_bias_test for i in range(bins-1)]
    bin_probs_test.append(_df_test["target"][bin_list[bins-1]:].mean())
    
    plt.scatter(bin_labels, bin_probs_train, label="Training Set", s=80, alpha=0.4)
    plt.scatter(bin_labels, bin_probs_test, label="Test Set", s=20, c='black')
    plt.hlines(1, 0, bins, linestyles="dotted")
    #plt.text(x=0.7*len(_df_train), y=0.02+max_bias, s="mean probability")
    plt.title("Lift Chart for P(target=1)")
    plt.xlabel(f"Sorted+Binned Prediction Probabilities, {bins} bins")
    plt.ylabel(f"Lift")
    plt.legend()
    return plt.show()


#------------------------------------------------
#------------------------------------------------


def cost_plot(model_, X_test_, y_test_, false_pos_cost_col=1, false_neg_cost_col=1):
    """
    PARAMETERS
    ----------
    model_: a fitted binary choice model that is accessible with scikit-learn-style methods.
    X_test_: pandas DataFrame, test/validation feature matrix.
    y_test_: list/1D-array/series, test target vector.
    false_pos_cost_col: str or numeric. 
        str: the name of the column in the feature matrix that contains the cost of each
            false positive instance (e.g. could be amount loaned in case of default)
        numeric: a fixed value for the cost of a false positive.
    false_neg_cost_col: str or numeric. 
        str: the name of the column in the feature matrix that contains the cost of each
            false negative instance (e.g. could be amount loaned in case of default)
        numeric: a fixed value for the cost of a false negative.
        
    WHAT THIS DOES
    --------------
    scikit-learn's predict_proba() method returns a vector of probabilities for "success" 
    (1 or "default" in loan case) from a model object when given a feature matrix's values.
    We define the "cutoff" as the probability where we consider a point classified as 1.
    This function plots the costs of misclassification error vs cutoffs (in 0.1% increments)
    as a scatterplot
    
    RETURNS
    -------
    Nada.
    """

    cutoff_ = []
    recall_ = []
    accuracy_ = []
    cost_of_errors_ = []
    
    df_co_ = pd.DataFrame()
    df_co_["target"] = y_test_.copy()
        
    
    dtrain_predictions = model_.predict(X_test_)
    dtrain_predprob = model_.predict_proba(X_test_)[:,1]

    for co_ in np.arange(0, .99, 0.01):
        cutoff_.append(co_)
        
        df_co_["preds"] = np.where(dtrain_predprob>co_, 1, 0)
        df_co_["is_wrong"] = np.where(df_co_.preds != df_co_.target, 1, 0)
        df_co_["false_pos"] = np.where((df_co_.target==0) & (df_co_.is_wrong==1), 1, 0)   
        df_co_["false_neg"] = np.where((df_co_.target==1) & (df_co_.is_wrong==1), 1, 0)
        
        if type(false_pos_cost_col) == str:
            coe_ = ((df_co_.false_pos*X_test_[false_pos_cost_col]).sum()*0.1 + 
                                   (df_co_.false_neg*X_test_[false_neg_cost_col]).sum()*0.9)
            cost_of_errors_.append(coe_/1000000)
        else:
            cost_of_errors_.append((df_co_.false_pos*np.ones(y_test_.shape)).sum()*false_pos_cost_col*0.1 + 
                                   (df_co_.false_neg*np.ones(y_test_.shape)).sum()*false_neg_cost_col*0.9)
        cm_ = metrics.confusion_matrix(y_test_, df_co_.preds)
        recall_.append(((cm_[1][1])/(cm_[1][0]+cm_[1][1])).round(3))
        accuracy_.append(((cm_[0][0]+cm_[1][1])/(cm_[0][0]+cm_[0][1]+cm_[1][0]+cm_[1][1])).round(3))
        
    df_graph = pd.DataFrame(list(zip(cutoff_, recall_, accuracy_, cost_of_errors_)), 
                        columns=["cutoff", "recall", "accuracy", "cost_of_errors"])

    print(f"Max Accuracy cutoff: {(df_graph.sort_values(by='accuracy', ascending=False).cutoff.iloc[0]).round(2)}")
    print(f"Min cost cutoff: {(df_graph.sort_values(by='cost_of_errors', ascending=True).cutoff.iloc[0]).round(2)}")
    
    plt.scatter(x="cutoff", y="cost_of_errors", data=df_graph, s=4, c="red")
    plt.ylabel("Loss from Incorrect Model Predictions ($millions)")
    plt.xlabel("Model-Predicted Probability At Which We Assume Defaults")
    plt.title("Cost Curve by Cutoff Point")
    plt.show()


#------------------------------------------------
#------------------------------------------------


def recall_plot(model_, X_test_, y_test_):
    """
    PARAMETERS
    ----------
    model_: a fitted binary choice model that is accessible with scikit-learn-style methods.
    X_test_: pandas DataFrame, test/validation feature matrix.
    y_test_: list/1D-array/series, test target vector.
    
        
    WHAT THIS DOES
    --------------
    scikit-learn's predict_proba() method returns a vector of probabilities for "success" 
    (1 or "default" in loan case) from a model object when given a feature matrix's values.
    We define the "cutoff" as the probability where we consider a point classified as 1.
    This function plots a model's recall and accuracy.
    
    RETURNS
    -------
    Nada.
    """

    cutoff_ = []
    recall_ = []
    accuracy_ = []
    
    df_co_ = pd.DataFrame()
    df_co_["target"] = y_test_.copy()
        
    
    dtrain_predictions = model_.predict(X_test_)
    dtrain_predprob = model_.predict_proba(X_test_)[:,1]

    for co_ in np.arange(0, .99, 0.01):
        df_co_["preds"] = np.where(dtrain_predprob>co_, 1, 0)
        cutoff_.append(co_)
        cm_ = metrics.confusion_matrix(y_test_, df_co_.preds)
        recall_.append(((cm_[1][1])/(cm_[1][0]+cm_[1][1])).round(3))
        accuracy_.append(((cm_[0][0]+cm_[1][1])/(cm_[0][0]+cm_[0][1]+cm_[1][0]+cm_[1][1])).round(3))
        
    df_graph = pd.DataFrame(list(zip(cutoff_, recall_, accuracy_)), 
                        columns=["cutoff", "recall", "accuracy"])

    print(f"Max Accuracy cutoff: {(df_graph.sort_values(by='accuracy', ascending=False).cutoff.iloc[0]).round(2)}")
    # print(f"Max Recall cutoff: {(df_graph.sort_values(by='recall', ascending=False).cutoff.iloc[0]).round(2)}")
    plt.scatter(x="cutoff", y="recall", data=df_graph, s=4, label="Proportion of Trues Correctly Classified")
    plt.scatter(x="cutoff", y="accuracy", data=df_graph, s=4, c="red", label="Proportion of All Data Correctly Classified")
    plt.xlabel("Cutoff Percentage")
    plt.ylabel("Proportion")
    plt.legend()
    plt.title(f"Recall and Accuracy for target")
    plt.ylim(0, 1.2)
    plt.xlim(-0.05, 1)
    plt.show()

    plt.show()


