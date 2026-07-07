\# Complete Data Cleaning in Python — Customer Sales Dataset



A full concept-to-code walkthrough of professional data cleaning: for every step, the underlying concept is explained first (why it matters, when to use it, what mistake to avoid), followed by the exact code used to implement it on a real messy dataset.



\## Table of Contents

\- \[Overview](#overview)

\- \[Problem](#problem)

\- \[Dataset](#dataset)

\- \[Tools \& Tech](#tools--tech)

\- \[Data Pipeline](#data-pipeline)

\- \[Skills Demonstrated](#skills-demonstrated)

\- \[Walkthrough: Concept + Code](#walkthrough-concept--code)

&#x20; - \[Step 1: Data Collection \& Inspection](#step-1-data-collection--inspection)

&#x20; - \[Step 2: Handling Missing Values](#step-2-handling-missing-values)

&#x20; - \[Step 3: Fixing Inconsistent Formatting](#step-3-fixing-inconsistent-formatting)

&#x20; - \[Step 4: Handling Duplicates](#step-4-handling-duplicates)

&#x20; - \[Step 5: Correcting Data Types](#step-5-correcting-data-types)

&#x20; - \[Step 6: Handling Outliers](#step-6-handling-outliers)

\- \[How to Run](#how-to-run)

\- \[Results](#results)

\- \[Future Work](#future-work)

\- \[Author \& Contact](#author--contact)



\## Overview

In interviews, the common questions are: \*"How do you handle messy data?"\*, \*"What steps do you take for data cleaning?"\*, and \*"How do you check if your data is clean?"\* The easy answer — "check for missing and duplicate values, then drop them" — doesn't work in a real company, where you can't remove a record or column without a strong, explainable reason.



This project documents the complete, defensible data cleaning process: \*\*why\*\* raw data gets messy in the first place, \*\*how\*\* to inspect and identify every category of problem before fixing anything, and \*\*how\*\* to correct each problem using the technique that fits that specific column and situation — not a blanket `dropna()`.



\## Problem

Company data is collected from multiple sources — web scraping, existing databases, APIs, surveys — and merged into a single data warehouse. When this raw data is pulled for analysis, it typically has:

\- \*\*Missing values\*\* — a field wasn't captured during collection

\- \*\*Duplicate values\*\* — the same record entered more than once (exact or near-duplicate due to formatting)

\- \*\*Spelling errors\*\* — manual form entry introduces typos (e.g., misspelled city names)

\- \*\*Outliers\*\* — extreme values, e.g. a negative price, or a value far outside the normal range

\- \*\*Incorrect data types\*\* — e.g. a date stored as text, or a numeric column stored as an object because of mixed formatting

\- \*\*Inconsistent formatting\*\* — the same real-world value written in different ways (`Male`/`M`, `Bengaluru`/`bengaluru `)



If these aren't fixed, every downstream step — analysis, machine learning, deep learning, dashboards — produces wrong conclusions, because the input itself is broken. \*\*Data cleaning\*\* is the process of detecting, correcting, and handling these errors so raw data becomes fit for analysis and modeling.



\## Dataset

A \*\*customer sales dataset\*\*, `messy\_customer\_sales\_data.csv`, with \*\*10,200 rows and 12 columns\*\*:



| Column | Description |

|---|---|

| Customer\_ID | Primary key (unique customer identifier) |

| Name | Customer name |

| Gender | Inconsistent formatting: `M`, `F`, `Male`, `FEMALE`, `female`, with stray spaces |

| Age | Mixed format: plain numbers and strings like `"52.0 years"`; contains outliers |

| City | Mixed casing and leading/trailing spaces (`" KOLKATA "`, `"bangalore"`) |

| Signup\_Date | Stored as text, needs conversion to datetime |

| Last\_Purchase\_Date | Stored as text, needs conversion to datetime |

| Purchase\_Amount | Numeric; contains missing values, a negative value, and extreme highs |

| Feedback\_Score | Numeric 1–10, behaves like a discrete/categorical value |

| Email | Text, no missing values |

| Phone\_Number | Integer, but treated as categorical (each customer has a different number) |

| Country | Same value ("India") written 4 different ways: `India`, `india`, `InDia`, `IND` |



\## Tools \& Tech

\- Python

\- pandas

\- NumPy

\- Matplotlib \& Seaborn (box plots for outlier visualization)

\- SciPy (`scipy.stats.zscore` for outlier detection)

\- Regex (`re`) for extracting numeric values from mixed-format strings

\- Jupyter Notebook



\## Data Pipeline

```

Raw CSV (10,200 rows, 12 cols)

&#x20;  → Step 1: Inspection (dtypes, describe, isnull, duplicated, value\_counts)

&#x20;  → Step 2: Handle Missing Values (drop key-missing rows, impute rest by type)

&#x20;  → Step 3: Fix Inconsistent Formatting (case, whitespace, category mapping)

&#x20;  → Step 4: Remove Duplicates (exact rows + primary-key near-duplicates)

&#x20;  → Step 5: Correct Data Types (object → int64, object → datetime)

&#x20;  → Step 6: Handle Outliers (Z-score method on Age \& Purchase\_Amount)

&#x20;  → Clean Dataset (df\_clean, analysis-ready)

```



\## Skills Demonstrated

\- Structured, non-destructive data inspection before making any changes

\- Deciding \*when\* to drop vs. impute (the 50%-missing rule)

\- Choosing the right imputation strategy per data type: median/mode for numeric, mode for categorical, forward-fill for time series

\- Regex-based extraction of numeric values from mixed-format text

\- Writing reusable functions instead of repeating logic

\- String cleaning with `.str.lower()`, `.str.strip()`, and dictionary-based `.replace()`

\- Detecting near-duplicate primary keys caused purely by formatting inconsistency

\- `drop\_duplicates()` at both row level and primary-key level

\- Data type correction: object → `int64`, object → `datetime`

\- Statistical outlier detection using Z-score, validated visually with box plots

\- Distinguishing genuine outliers (domain-valid extremes) from data-entry errors



\## Walkthrough: Concept + Code



\### Step 1: Data Collection \& Inspection



\*\*Concept:\*\*

Before fixing anything, you inspect and explore the data to identify every problem it has — this is effectively EDA focused on data quality. Skipping a thorough inspection means discovering problems mid-analysis, which wastes time. First check the data type expected in each column (numeric, categorical, date-time, text), then check for missing values, duplicate values, and inconsistencies. Useful techniques: summary statistics (`describe()`) to spot outliers and understand distribution, `info()` to see data types and non-null counts, and value counts / unique values for categorical columns to spot formatting problems.



\*\*Code:\*\*

```python

import pandas as pd

import matplotlib.pyplot as plt



df = pd.read\_csv('messy\_customer\_sales\_data.csv')

df

```

Loads the raw data: 10,200 rows, 12 columns.



```python

df.info()

```

\*\*Finding:\*\* `Age` should be numeric but is stored as an object (because some values include `"years"`). `Signup\_Date` and `Last\_Purchase\_Date` should be date-time but are stored as objects.



```python

df.describe().round()

```

\*\*Finding:\*\* `Purchase\_Amount` has a negative minimum value (invalid — a price can't be negative) and a maximum value far above the 75th percentile — both are outliers.



```python

df.isnull().sum()          # count of missing values per column

df.shape                    # confirm total records

df.isnull().sum() / len(df) \* 100   # % missing per column

```

This shows which columns have missing data and how significant it is proportionally — some columns are missing \~9-10% of values.



```python

df.dropna().shape

```

Shows what happens if you naively drop every row with \*\*any\*\* missing value: the dataset collapses from 10,200 rows to a much smaller number — roughly a 60% data loss. This is why a blanket `dropna()` is the wrong first move.



```python

df\[df.duplicated()].shape

df\[df.duplicated()]

```

Finds fully duplicated rows (15 found on the first pass, before formatting fixes).



```python

df\['Customer\_ID'].value\_counts().head(10)

```

Since `Customer\_ID` is the primary key, it should never repeat. This surfaces customers whose ID appears more than once.



```python

df\[df\['Customer\_ID'] == 'CUST3374']   # city spelling difference

df\[df\['Customer\_ID'] == 'CUST7749']   # gender spelling difference

```

Investigating these "duplicate" primary keys shows they weren't caught by `df.duplicated()` because a formatting difference (e.g. `Bengaluru` vs `bengaluru `, or `Female` vs `F`) made pandas treat them as two distinct rows, even though they represent the same customer.



```python

for col in df.columns:

&#x20;   if df\[col].nunique() < 20:

&#x20;       print(df\[col].value\_counts())

&#x20;       print('-'\*50)

```

Loops through every column and, for any column with fewer than 20 unique values (i.e. categorical), prints its value counts. This is how every formatting inconsistency was found at once: `Gender` had 6 variants instead of 2, `City` had duplicated names differing only by case/spacing, and `Country` had 4 variants of `India`.



\### Step 2: Handling Missing Values



\*\*Concept:\*\*

Two options: delete or impute (fill in). Delete a row or column only when \*\*more than 50%\*\* of its data is missing — imputing that much data would produce misleading, over-fabricated conclusions. Otherwise, impute:

\- \*\*Numeric columns:\*\* mean (no outliers), median (outliers present), or mode (discrete values)

\- \*\*Categorical columns:\*\* mode, or a new category like `"Unknown"`/`"Other"` if no value can be reasonably guessed (e.g. email, phone number)

\- \*\*Time series / date-time columns:\*\* forward-fill or backward-fill

\- \*\*Advanced case:\*\* predictive imputation — fit a machine learning model on the non-missing data to predict the missing values (resource-intensive, used when accuracy really matters)



The primary key (`Customer\_ID`) is the one exception: it can never be imputed, since there's no valid value to guess — rows missing it are dropped.



\*\*Code:\*\*

```python

df.dropna(subset=\['Customer\_ID'], inplace=True)   # Primary key should be unique

df

df.isnull().sum()

```

Drops only rows missing `Customer\_ID` — a small, defensible loss instead of losing 60% of the data.



```python

df\['Age'].unique()

```

Reveals values like `"52.0 years"` mixed with plain numbers — regex extraction is needed before any numeric operation (including median) can work.



```python

import re 



def extract\_age(age):

&#x20;   age\_num = re.findall('\[0-9]+', str(age))

&#x20;   if len(age\_num) > 0:

&#x20;       return age\_num\[0]

&#x20;   else:

&#x20;       return age



df\['Age'] = df\['Age'].apply(lambda x: extract\_age(x))

df\['Age'].unique()

```

A reusable function extracts the numeric portion from every value, regardless of whether it had `"years"` attached. Values with no digits at all (`NaN`) pass through unchanged for now.



```python

df\_age = df\[df\['Age'] != 'nan years']\['Age']

age\_median = int(df\_age.dropna().astype('int64').median())

age\_median

```

Calculates the median only from valid, present age values (excluding the residual `'nan years'` string and true `NaN`s), after casting to `int64`.



```python

df.replace('nan years', age\_median, inplace=True)

df\['Age'].unique()



df\['Age'].fillna(age\_median, inplace=True)

df\['Age'].unique()

```

Replaces the leftover `'nan years'` string and any remaining true `NaN` values with the computed median.



```python

df\['Purchase\_Amount'].fillna(df\['Purchase\_Amount'].median(), inplace=True)

df.isnull().sum()

```

`Purchase\_Amount` is already numeric, so its median can be used directly.



```python

df\['Feedback\_Score'].fillna(df\['Feedback\_Score'].mode()\[0], inplace=True)

df.isnull().sum()

```

`Feedback\_Score` behaves like a discrete/categorical value (1–10), so mode is used instead of mean/median.



```python

for col in \['Gender', 'City', 'Country']:

&#x20;   df\[col].fillna(df\[col].mode()\[0], inplace=True)

```

A loop fills all three categorical columns with their respective mode in one pass, instead of writing the same line three times.



```python

df\['Last\_Purchase\_Date'].ffill(inplace=True)

```

`Last\_Purchase\_Date` is a date-time column, so forward-fill (assume similar to the most recent known value) is the appropriate method — after this, `df.isnull().sum()` returns all zeros.



\### Step 3: Fixing Inconsistent Formatting



\*\*Concept:\*\*

The same real-world value can appear in the data in multiple written forms — different casing (`Male` vs `MALE`), extra whitespace, or abbreviations (`M` vs `Male`). Logically these should collapse into one category. Fixes: convert to lowercase (or uppercase/title case, by preference), strip leading/trailing spaces, then use `.replace()` with a mapping dictionary to standardize abbreviations into full, consistent category names.



\*\*Code:\*\*

```python

df\['Gender'].unique()

```

Shows 6 inconsistent variants: mixed case, abbreviations, and stray spaces.



```python

df\['Gender'] = df\['Gender'].str.lower().str.strip()

df\['Gender'].replace({'m': 'male', 'f': 'female'}, inplace=True)

df.head()

```

Lowercasing and stripping first collapses case/whitespace variants down to 4 categories; the replace-dictionary then maps `m`/`f` to `male`/`female`, leaving exactly 2 correct categories.



```python

df\['City'].unique()

df\['City'] = df\['City'].str.strip().str.lower()

```

Same fix applied to `City` — strips whitespace and lowercases so `" KOLKATA "` and `"kolkata"` become one category.



```python

df\['Country'] = df\['Country'].str.lower()

df\['Country'].replace({'ind': 'India'}, inplace=True)

df\['Country'].unique()

df\['Country'].replace({'india': 'India'}, inplace=True)

df.head()

```

`Country` had 4 variants of "India" (`India`, `india`, `InDia`, `IND`). Lowercasing collapses 3 of them; the remaining abbreviation (`ind`) is mapped explicitly, then the lowercase result is mapped back to the proper display form `India`.



\### Step 4: Handling Duplicates



\*\*Concept:\*\*

Handle duplicates in two layers: first remove exact duplicate rows. Then, separately, check the primary key — if the same `Customer\_ID` appears more than once (even if the rows aren't 100% identical, e.g. differing only by a date field), keep only one record and drop the rest, since a primary key must be unique. It's worth re-checking duplicate counts \*after\* fixing formatting — inconsistent casing can hide duplicates that only become visible once values are standardized.



\*\*Code:\*\*

```python

df\[df.duplicated()]

df\[df.duplicated()].shape

```

Duplicate count is checked again post-formatting-fix — it grew from the original 15 to a much larger number (163), because standardizing casing revealed rows that were actually identical but previously looked different.



```python

df.drop\_duplicates(inplace=True)

```

Removes all exact duplicate rows.



```python

df\['Customer\_ID'].value\_counts()

df\[df\['Customer\_ID'] == 'CUST9534']

```

Even after removing exact duplicates, some `Customer\_ID`s still appear twice — these are near-duplicates (same customer, e.g. differing `Last\_Purchase\_Date`).



```python

df.drop\_duplicates(subset=\['Customer\_ID'], keep='first', inplace=True)

df.shape

```

Keeps only the first record for each `Customer\_ID`, guaranteeing the primary key is truly unique.



\### Step 5: Correcting Data Types



\*\*Concept:\*\*

Every column should hold the data type appropriate to its content: numeric columns as `int`/`float`, categorical columns as `object`, and date columns as `datetime`. Type corrections must happen \*\*after\*\* missing values and formatting are fixed — attempting to cast a column with `NaN`s or mixed-format text (like `"years"` suffixes) directly to a numeric or date type raises an error.



\*\*Code:\*\*

```python

df\['Age'] = df\['Age'].astype('int64')

df.columns

```

`Age` is now purely numeric strings after Step 2, so it converts cleanly to `int64`.



```python

df\['Signup\_Date'] = pd.to\_datetime(df\['Signup\_Date'])

df\['Last\_Purchase\_Date'] = pd.to\_datetime(df\['Last\_Purchase\_Date'])

df.dtypes

```

Both date columns convert from object to proper `datetime64` type using `pd.to\_datetime()`.



\### Step 6: Handling Outliers



\*\*Concept:\*\*

Not every outlier should be removed — a "genuine" outlier (e.g. a legitimately expensive luxury house in a price dataset) is a valid, domain-explainable extreme value. But an outlier caused by a data-entry error (like a negative purchase amount) is not genuine and should be removed. Outliers can be detected using the interquartile range (IQR) method or the \*\*Z-score\*\* method — a Z-score greater than 3 standard deviations from the mean is generally treated as an outlier. Box plots are used to visually confirm which columns actually have outliers before applying any statistical filter.



\*\*Code:\*\*

```python

import matplotlib.pyplot as plt

import seaborn as sns



cols = \['Age', 'Purchase\_Amount', 'Feedback\_Score']

plt.figure(figsize=(15, 5))

for i, col in enumerate(cols, 1):

&#x20;   plt.subplot(1, 3, i)

&#x20;   sns.boxplot(y=df\[col], color='skyblue', width=0.3)

&#x20;   plt.title(f'Boxplot of {col}', fontsize=12)

&#x20;   plt.ylabel('')

&#x20;   plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight\_layout()

plt.show()

```

Box plots show outlier points (dots beyond the whiskers) present in `Age` and `Purchase\_Amount`, but none in `Feedback\_Score`.



```python

from scipy import stats

stats.zscore(df\[\['Age', 'Purchase\_Amount']])

```

Computes the Z-score for every value in both columns — how many standard deviations each value is from the column mean.



```python

import numpy as np

z\_score = np.abs(stats.zscore(df\[\['Age', 'Purchase\_Amount']]))

```

Takes the absolute value so both large positive and large negative deviations are treated the same way.



```python

df\_clean = df\[\~(z\_score > 3).any(axis=1)]

df\_clean.shape

```

Filters out any row where either `Age` or `Purchase\_Amount` has a Z-score greater than 3 — these are the statistical outliers, including the impossible negative purchase amount and the improbably high age (250+) that surfaced during inspection.



```python

plt.figure(figsize=(15, 5))

for i, col in enumerate(cols, 1):

&#x20;   plt.subplot(1, 3, i)

&#x20;   sns.boxplot(y=df\_clean\[col], color='skyblue', width=0.3)

&#x20;   plt.title(f'Boxplot of {col}', fontsize=12)

&#x20;   plt.ylabel('')

&#x20;   plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight\_layout()

plt.show()

```

Re-plotting the box plots on `df\_clean` confirms no outliers remain in `Age` or `Purchase\_Amount`.



\## How to Run

1\. Clone this repository and open `data\_Cleaning\_Process.ipynb` in Jupyter.

2\. Install dependencies:

&#x20;  ```

&#x20;  pip install pandas numpy matplotlib seaborn scipy

&#x20;  ```

3\. Place `messy\_customer\_sales\_data.csv` in the same working directory as the notebook.

4\. Run all cells top to bottom, in order — inspection → missing values → formatting → duplicates → data types → outliers.

5\. Export the final cleaned DataFrame if needed:

&#x20;  ```python

&#x20;  df\_clean.to\_csv('cleaned\_customer\_sales\_data.csv', index=False)

&#x20;  ```



\## Results

\- Started with \*\*10,200 rows, 12 columns\*\* of raw data.

\- After dropping rows with a missing primary key: reduced to a smaller, still-substantial dataset (avoiding the \~60% loss a naive `dropna()` would have caused).

\- After removing exact duplicates (163 rows) and near-duplicate primary keys: data further reduced to unique, one-record-per-customer rows.

\- After Z-score-based outlier removal on `Age` and `Purchase\_Amount`: final `df\_clean` has zero missing values, zero duplicates, consistent formatting, correct data types, and no statistical outliers in either `Age` or `Purchase\_Amount`.

\- Every row removed has a specific, explainable reason — missing primary key, exact duplicate, near-duplicate primary key, or statistical outlier — which is exactly the justification expected when explaining data loss to a manager or in an interview.



\## Future Work

\- Extend to predictive imputation (fit a model on non-missing features to impute missing values) instead of median/mode where higher accuracy is needed.

\- Add automated data quality checks (e.g. Great Expectations) to catch these issues at ingestion time in future pipelines.

\- Package the full cleaning sequence into a reusable function/pipeline for use across similar datasets.



\## Author \& Contact

\- \*\*Name:\*\* Sahajahanur Rahman

\- \*\*Email:\*\* connectingsrl@gmail.com

\- \*\*GitHub:\*\* \[Sahajahanur](https://github.com/Sahajahanur)

