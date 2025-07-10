#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import TimeSeriesSplit


# In[2]:


df_raw = pd.read_csv('./ncd1.csv',header=None)
df = df_raw.iloc[4:, :].copy()
df.columns = [
    'date',
    "reverse_repo_7d",
    "reverse_repo_7d_amount",
    "MLF_1y",
    "DXY",
    "DR001",
    "DR007",
    "FR007",
    "CPI_yoy",
    "CPI_mom",
    'CPI',
    "PPI_yoy",
    "PPI_mom",
    "house_listing_index",
    "house_listing_index2",
    "PMI",
    "trust_loan",
    "m1_yoy",
    "m2_yoy",
    "tsf_yoy",
    "Shibor",
    "central_bank_bill",
    "central_bank_bill_amount",
    "ncd_amount",
    "ncd"
]

cols_to_drop = [col for col in df.columns if any(key in col for key in ['DR007', 'MLF_1y', 'DR001'])]
df = df.drop(columns=cols_to_drop)

# 清洗数字列：去掉逗号、百分号、中文单位等非数字字符
for col in df.columns:
    if col != "date":
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)   # 去除千位逗号
            .str.replace("%", "", regex=False)   # 去除百分号（如有）
            .str.replace("亿", "", regex=False)  # 去除“亿”字（如有）
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 按日期升序排序
df = df.sort_values(by='date', ascending=True).reset_index(drop=True)

# 转换为datetime
df["date"] = pd.to_datetime(df["date"])
df.to_csv('ncd1_sorted.csv', index=False)

# 重置索引
df.reset_index(drop=True, inplace=True)


# In[3]:


df.head(50)


# In[4]:


df.iloc[8000:8100]


# In[5]:


df.isna().sum()


# In[6]:


# 根据收益率计算其他因子
## cntn
df["ncd"] = pd.to_numeric(df["ncd"], errors="coerce")
# 收益率差分
df["ncd_diff"] = df["ncd"].diff()
# cntn(30): 最近30天收益率变化<0的天数占比
df["cntn_30"] = (
    df["ncd_diff"]
    .rolling(30)
    .apply(lambda x: (x < 0).mean(), raw=True)
)


# In[7]:


# df = df.drop(columns=["IRS_FR007_7d", "tsf_yoy"])
df.head(50)


# In[8]:


# 去除所有含NaN的行
data = df.dropna().copy()
# 重置索引
data.reset_index(drop=True, inplace=True)
data.head()


# In[9]:


data.isna().sum()


# In[10]:


# 要标准化的列
factor_cols = [col for col in data.columns if col not in ["date", "ncd", "ncd_diff"]]

# 初始化scaler
scaler = StandardScaler()

# 标准化
scaled_array = scaler.fit_transform(data[factor_cols])

# 转回DataFrame
scaled_df = pd.DataFrame(
    scaled_array,
    columns=factor_cols,
    index=data.index
)

# 把日期和目标收益率拼回去
scaled_df["date"] = data["date"]
scaled_df["ncd"] = data["ncd"]
scaled_df["ncd_diff"] = data["ncd_diff"]

# 查看结果
scaled_df.head(50)


# In[11]:


# 创建1周后的收益率
scaled_df["ncd_next_week"] = scaled_df["ncd"].shift(-7)
scaled_df.head(100)


# In[12]:


# 绘图
for col in df.columns:
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df[col])
    plt.title(f"{col} over time")
    plt.xlabel("Date")
    plt.ylabel(col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[13]:


# 相关性检验
feature_cols = [col for col in scaled_df.columns if col not in ["date", "ncd", "ncd_diff"]]

# Pearson
pearson_corr = scaled_df[feature_cols].corr()["ncd_next_week"]

# Spearman
spearman_corr = scaled_df[feature_cols].corr(method="spearman")["ncd_next_week"]

print("Pearson相关系数：")
print(pearson_corr)

print("\nSpearman相关系数：")
print(spearman_corr)

print('*' * 100)

# 阈值
threshold = 0.05

# 筛选
selected_pearson = pearson_corr[pearson_corr.abs() > threshold].index.tolist()
selected_spearman = spearman_corr[spearman_corr.abs() > threshold].index.tolist()

# 去掉ncd_tomorrow本身
# selected_pearson = [f for f in selected_pearson if f != "ncd_next_week"]
# selected_spearman = [f for f in selected_spearman if f != "ncd_next_week"]

print("Pearson筛选出的因子：", selected_pearson)
print("Spearman筛选出的因子：", selected_spearman)
print('*' * 100)

# 取交集
final_factors = list(set(selected_pearson) & set(selected_spearman))
print("最终筛选因子：", final_factors)


# In[14]:


# 计算Spearman IC
from scipy.stats import spearmanr

print("未来1周IC：")
for f in feature_cols:
    valid_idx = scaled_df[[f, "ncd_next_week"]].dropna().index
    ic = spearmanr(
        scaled_df.loc[valid_idx, f],
        scaled_df.loc[valid_idx, "ncd_next_week"]
    ).correlation
    print(f"{f}: IC = {ic:.4f}")


# In[15]:


# 提取 IC值筛选过的因子DataFrame
X_final = scaled_df[final_factors]

# 计算相关矩阵
corr_matrix = X_final.corr()

# 打印
print(corr_matrix)


# In[16]:


# 计算相关矩阵（可选，如果已有就跳过）
# corr_matrix = X_final.corr()

# 存储高相关因子对
high_corr_pairs = []

print("\n以下因子对相关系数 > 0.8：")
for i in range(len(final_factors)):
    for j in range(i+1, len(final_factors)):
        f1 = final_factors[i]
        f2 = final_factors[j]
        corr_ij = corr_matrix.loc[f1, f2]
        if abs(corr_ij) > 0.8:
            print(f"  {f1} 与 {f2} 的相关系数 = {corr_ij:.4f}")
            high_corr_pairs.append((f1, f2, corr_ij))

# 可选：将结果保存为 DataFrame 以便导出或排序
high_corr_df = pd.DataFrame(high_corr_pairs, columns=["factor_1", "factor_2", "correlation"])


# In[17]:


# 创建布尔掩码
mask = np.ones(len(final_factors), dtype=bool)

print("\n以下因子对共线性>0.8：")
for i in range(len(final_factors)):
    for j in range(i+1, len(final_factors)):
        corr_ij = corr_matrix.iloc[i, j]
        if abs(corr_ij) > 0.8:
            f1 = final_factors[i]
            f2 = final_factors[j]
            print(f"  {f1} 与 {f2} 相关系数 = {corr_ij:.4f}")
            mask[j] = False  # 依旧去掉第j个

# 根据mask过滤
final_factors_no_collinearity = list(np.array(final_factors)[mask])

final_factors_no_collinearity = [
    f for f in final_factors_no_collinearity
    if f not in ["ncd_next_week", "ncd_label"]
]

print("\n自动去除共线性后且不含目标变量的因子：", final_factors_no_collinearity)


# In[18]:


# 差值
scaled_df["ncd_diff_next_week"] = scaled_df["ncd_next_week"] - scaled_df["ncd"]

# 阈值
epsilon = 0

# 分类标签
def classify(x):
    if x > epsilon:
        return "up"
    elif x < -epsilon:
        return "down"

scaled_df["ncd_label"] = scaled_df["ncd_diff_next_week"].apply(classify)

# 看分布
print(scaled_df["ncd_label"].value_counts())

# 去除Na值
scaled_df = scaled_df.dropna().reset_index(drop=True)


# In[19]:


scaled_df.head()


# In[20]:


class Para:
    percent_cv = 0.2
    percent_train = 0.8
    seed = 42
    ma_window = 5


# In[23]:


# x和y
x = scaled_df[final_factors_no_collinearity]
y = scaled_df["ncd_label"]

# 按行数切分
train_size = int(len(scaled_df) * Para.percent_train)

x_train = x.iloc[:train_size]
x_test = x.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]


# In[25]:


model = RandomForestClassifier(random_state = Para.seed)
params = {
    'n_estimators': [100,200,300],
    'max_depth': [2,4,6,8]
}
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=tscv
)
grid_search.fit(x_train,y_train)
print(f'score is {grid_search.score(x_train,y_train)}')
print(f'best parameters are {grid_search.best_params_}')


# In[26]:


model = RandomForestClassifier(n_estimators = 100, max_depth = 4, random_state = Para.seed)
model.fit(x_train,y_train)


# In[27]:


predictions = model.predict(x_test) 


# In[28]:


y_pred = grid_search.predict(x_test)
print("=== 测试集性能 ===")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[29]:


comparison = pd.DataFrame({
    "y_test": y_test.values,
    "y_pred": y_pred
})

print(comparison.head(50))


# ## Gradient Boosting

# In[ ]:


'''
# 1. 定义模型
gb_model = GradientBoostingClassifier(random_state=42)

# 2. 设置参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [3, 4, 5]
}

# 3. 网格搜索交叉验证
grid_search = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

# 4. 训练模型
grid_search.fit(x_train, y_train)

# 5. 输出最优参数
print("Best parameters:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)

# 6. 用最优模型预测
y_pred = grid_search.predict(x_test)

# 7. 输出评估指标
print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
'''


# In[ ]:


'''
import matplotlib.pyplot as plt

feature_importances = grid_search.best_estimator_.feature_importances_

plt.figure(figsize=(10,6))
plt.barh(x_train.columns, feature_importances)
plt.title("Feature Importances")
plt.show()
'''


# In[ ]:




