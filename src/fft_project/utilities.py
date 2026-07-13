import logging

logger = logging.getLogger(__name__)

def Gini_impurity_subset(df, true_class_column_name, classes):
  proportions = {}
  N = len(df)
  G=1.0
  for c in classes:
    p = len(df[df[true_class_column_name] == c]) / N
    proportions[c] = p
    G = G - p*p
  return G



def Gini_impurity_partition(df, assigned_class_column_name, true_class_column_name, classes, verbose=False):
  if verbose:
    logger.info(f"Calculating Gini impurity of {assigned_class_column_name} classification with reference to {true_class_column_name} labels")
  S={}
  G={}
  M = len(df)
  for assigned_class in classes:
    S[assigned_class] = df[df[assigned_class_column_name]==assigned_class]
    G[assigned_class] = Gini_impurity_subset(S[assigned_class], true_class_column_name, classes)
    N = len(S[assigned_class])
    if verbose:
      logger.info(f"{N} items assigned to class {assigned_class}")

    for c in classes:
      p = len(S[assigned_class][S[assigned_class][true_class_column_name] == c])/N
      if verbose:
        logger.info(f"Proportion {p} of these have label {c}")
  G_split=0.0
  for c in classes:
    G_split = G_split + (len(S[c])/M) * G[c]

  return G_split

def confusion_rates(df, assigned_class_column_name, true_class_column_name, target_class, verbose=False):
  N=len(df)
  TP = len(df[(df[assigned_class_column_name] == target_class) & (df[true_class_column_name] == target_class)])
  FP = len(df[(df[assigned_class_column_name] == target_class) & (df[true_class_column_name] != target_class)])
  TN = len(df[(df[assigned_class_column_name] != target_class) & (df[true_class_column_name] != target_class)])
  FN = len(df[(df[assigned_class_column_name] != target_class) & (df[true_class_column_name] == target_class)])
  if TP+FP + TN + FN != N:
    logger.error(f"Counts do not sum to total number of rows: {TP+FP + TN + FN} != {N}")
    raise ValueError(f"Counts do not sum to total number of rows: {TP+FP + TN + FN} != {N}")
  TPR = TP/(TP+FN)
  FPR = FP/(FP+TN)
  TNR = TN/(TN+FP)
  FNR = FN/(FN+TP)
  if verbose:
    logger.info(f"Confusion rates for target class {target_class}: TPR={TPR}, FPR={FPR}, TNR={TNR}, FNR={FNR}")
  return TPR, FPR, TNR, FNR

