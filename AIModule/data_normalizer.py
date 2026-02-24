import os
import pandas as pd
import kagglehub

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

norm_ds1 = True
norm_ds2 = False
norm_ds3 = False
grade_range = (1, 5)


# https://zenodo.org/records/16623461
def save_dataset1():

    df = pd.read_csv("StudentGradesDataset.csv")

    meta_cols = ["Serial No", "School Name", "Class", "Years"]

    subject_cols = [c for c in df.columns if c not in meta_cols]

    long_df = df.melt(
        id_vars=["Serial No", "Years"],
        value_vars=subject_cols,
        var_name="subject",
        value_name="grade_raw"
    )

    long_df["grade_raw"] = pd.to_numeric(long_df["grade_raw"], errors="coerce")
    long_df = long_df.dropna(subset=["grade_raw"])

    long_df.to_csv("dataset_temp.csv", index=False)

    def scale_subject(x):
        scaler = MinMaxScaler(feature_range=grade_range)
        return scaler.fit_transform(x.values.reshape(-1, 1)).ravel()

    long_df["grade_scaled"] = (
        long_df
        .groupby("subject")["grade_raw"]
        .transform(scale_subject)
    )

    long_df["year"] = long_df["Years"].str.extract(r"(\d{4})")

    wide_df = (
        long_df
        .pivot_table(
            index=["Serial No", "subject"],
            columns="year",
            values="grade_scaled"
        )
        .reset_index()
    )

    wide_df.columns = [
        f"grade_{c}" if str(c).isdigit() else c
        for c in wide_df.columns
    ]

    wide_df = wide_df.rename(columns={"Serial No": "student_id"})
    grade_cols = [c for c in wide_df.columns if c.startswith("grade_")]
    wide_df = (
        wide_df
        .assign(num_grades=lambda df: df[grade_cols].notna().sum(axis=1))
        .query("num_grades > 2")
        .drop(columns="num_grades")
    )
    print(wide_df.head())
    wide_df.to_csv("dataset1.csv", index=False)


# https://data.mendeley.com/datasets/6dgkv6kpr2/1
def save_dataset2():

    df = pd.read_excel("cgpaPRED.xls", sheet_name="Sheet1")

    df.insert(0, "student_id", range(1, len(df) + 1))

    df.insert(1, "subject", "cgpa")

    years = [2019, 2020, 2021, 2022, 2023]
    df.columns = ["student_id", "subject"] + [f"grade_{y}" for y in years]

    grade_cols = [f"grade_{y}" for y in years]

    scaler = MinMaxScaler(feature_range=grade_range)
    df[grade_cols] = scaler.fit_transform(df[grade_cols])

    df.to_csv("dataset2.csv", index=False)

    print(df.head())


# https://www.kaggle.com/datasets/dipam7/student-grade-prediction
def save_dataset3():
    df = pd.read_csv("student-mat.csv")

    grade_cols = ["G1", "G2", "G3"]
    df_grades = df[grade_cols].copy()

    df_grades.insert(0, "student_id", range(1, len(df_grades) + 1))

    df_grades.insert(1, "subject", "sgd")

    years = [2019, 2020, 2021]
    df_grades.columns = ["student_id", "subject"] + [f"grade_{y}" for y in years]

    scaler = MinMaxScaler(feature_range=grade_range)
    df_grades[[f"grade_{y}" for y in years]] = scaler.fit_transform(df_grades[[f"grade_{y}" for y in years]])

    df_grades.to_csv("dataset3.csv", index=False)

    print(df_grades.head())


if norm_ds1:
    save_dataset1()

if norm_ds2:
    save_dataset2()

if norm_ds3:
    save_dataset3()
