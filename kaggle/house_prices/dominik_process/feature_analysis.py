import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sklearn.linear_model as linear_model
import patsy
#import scipy.stats as stats

# set word dir
import os
from os import path
import sys

# os.listdir(".")
cur_dir = path.dirname("/data/code/github/machine_learn_msw/kaggle/house_prices/eda.py")
print(cur_dir)
sys.path.append(cur_dir)

# set pandas display options
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20

# get data
train = pd.read_csv(cur_dir + "/train.csv")
test = pd.read_csv(cur_dir + "/test.csv")


# 数据概况
def look_data(train):
    # take a look at data
    print(train.columns)
    print(train.shape)
    print(test.shape)
    print(train.head())

    for col in train.columns:
        print(train.dtypes[col])


def feature_nature(train):
    # 分别出标称型和数值型特征
    # 定性，标称型
    quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
    quantitative.remove('SalePrice')
    quantitative.remove('Id')
    # 定量，数值型
    qualitative = [f for f in train.columns if train.dtypes[f] == 'object']

    print(qualitative)
    print(quantitative)
    return quantitative, qualitative


quantitative, qualitative = feature_nature(train)


def show_missing_value(train):
    # 缺失值分析和处理
    # 获取各特征的缺失值情况
    missing = train.isnull().sum()
    print(missing)
    # 具有缺失值的特征
    true_missing = missing[missing > 0]
    print(true_missing)
    true_missing.sort_values(inplace=True)
    true_missing.plot.bar()
    plt.xticks(rotation=45)
    plt.show()


#show_missing_value(train)


def price_dist_show(train):
    # 房屋的价格最接近哪种概率分布
    y = train['SalePrice']
    plt.clf()
    plt.figure(1)
    plt.title('Johnson SU')
    sns.distplot(y, kde=True, fit=st.johnsonsu)
    plt.figure(2)
    plt.title('Normal')
    sns.distplot(y, kde=True, fit=st.norm)
    plt.figure(3)
    plt.title('Log Normal')
    sns.distplot(y, kde=True, fit=st.lognorm)
    plt.show()


# 根据结果看，Log Normal效果不错，最好的是Johnnson SU
# 如果使用线性回归，则最好对预测值进行对数操作，使其服从对数正态分布
#price_dist_show(train)


# 分析所有数值型特征的概率分布
def num_feature_dist_show(train):
    f = pd.melt(train, value_vars=quantitative)
    g = sns.FacetGrid(f, col="variable", col_wrap=3, sharex=False, sharey=False)

    #g = g.map(sns.distplot, "value", kde=True, fit=st.norm, fit_kws={'color':'r'})
    g = g.map(sns.distplot, "value", kde=True, fit=st.lognorm, fit_kws={'color':'r'})
    plt.show()

# 因为结果服从正态分布，所以各特征值也需要处理，选择可以转化为正态分布和对数正态分布的特征
# (线性回归是各特征之间的线性组合，所以结果服从各特征组合后的分布。对于正态分布，多个正态分布的线性组合还是正态分布。所以， 如果结果符合正态分布，则需要各特征也尽量服从正态分布)
# 如果特征已经较符合正态分布，则原值无需在处理；如果特征较符合对数正态分布，后续需要对其取对数
# log norm: LotFrontage, 1stFlrSF,
# norm: LotFrontage, LotArea, BsmtUnfSF, TotalBsmtSF, 1stFlrSF, GrLiveArea, GarageArea
#num_feature_dist_show(train)


def scalar_feature_fill_missing(train):
    for c in qualitative:
        train[c] = train[c].astype('category')
        if train[c].isnull().any():
            train[c] = train[c].cat.add_categories(['MISSING'])
            train[c] = train[c].fillna('MISSING')



# 分析所有标称型特征的概率分布
def scalar_feature_dist_show(train):
    scalar_feature_fill_missing(train)
    def boxplot(x, y, **kwargs):
        # 箱型图，统计学上常用 -- https://baike.baidu.com/item/%E7%AE%B1%E5%BD%A2%E5%9B%BE/10671164?fromtitle=%E7%AE%B1%E7%BA%BF%E5%9B%BE&fromid=10101649
        sns.boxplot(x=x, y=y)
        # 显示每个取值上对应的房屋售价
        # sns.swarmplot(x=x, y=y, color='.25')
        x = plt.xticks(rotation=90)

    f = pd.melt(train, id_vars=['SalePrice'], value_vars=qualitative)
    g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=4)
    g = g.map(boxplot, "value", "SalePrice")

    #plt.show()

scalar_feature_dist_show(train)


def feature_importance(train):
    # 在各标称型特种中，通过方差分析，如果p值越小，表明样本数据越具有总体代表性，我们这里认为其越重要。
    # (p值: 是将观察结果认为有效即具有总体代表性的犯错概率。如p=0.05提示样本中变量关联有5%的可能是由于偶然性造成的。许多的科学领域中产生p值的结果≤0.05被认为是统计学意义的边界线，但是这显著性水平还包含了相当高的犯错可能性。结果0.05≥p>0.01被认为是具有统计学意义，而0.01≥p≥0.001被认为具有高度统计学意义。)
    # 方差分析的f值即可作为衡量特征重要程度的一个参数
    def anova(frame):
        anv = pd.DataFrame()
        anv['feature'] = qualitative
        pvals = []
        for c in qualitative:
            samples = []
            for cls in frame[c].unique():
                s = frame[frame[c] == cls]['SalePrice'].values
                samples.append(s)
            pval = st.f_oneway(*samples)[1]
            pvals.append(pval)
        anv['pval'] = pvals
        return anv.sort_values('pval')

    a = anova(train)
    a['disparity'] = np.log(1. / a['pval'].values)
    sns.barplot(data=a, x='feature', y='disparity')
    plt.xticks(rotation=90, fontsize=9)
    plt.tight_layout(h_pad=1.08)  # 可以设置图边距
    plt.show()


#feature_importance(train)


# 对标称型特征进行编码
encode_map_list = {}
def encode_scalar_feature(train):
    # 按每个分类中所有房屋售价的平均值作为给定分类值的依据，均值越高，编码值越高
    def encode(frame, feature):
        ordering = pd.DataFrame()
        ordering['val'] = frame[feature].unique()
        ordering.index = ordering.val
        ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
        ordering = ordering.sort_values('spmean')
        ordering['ordering'] = range(1, ordering.shape[0] + 1)
        ordering = ordering['ordering'].to_dict()
        encode_map_list[feature] = ordering

        for cat, o in ordering.items():
            frame.loc[frame[feature] == cat, feature + '_E'] = o

    qual_encoded = []
    for q in qualitative:
        encode(train, q)
        qual_encoded.append(q + '_E')
    print(qual_encoded)
    return qual_encoded


qual_encoded = encode_scalar_feature(train)


def scalar_encode_by_list(map_list, data):
    for feature, encode_map in map_list.items():
        for cat, o in encode_map.items():
            data.loc[data[feature] == cat, feature + '_E'] = o


# 相关性分析
# 一般来说，为了减少混淆，只应将与其它变量不相关的变量加入回归模型（但需与销售价格相关）。
# 特征与售价的相关性。训练时应选择与售价最相关的一些特征，具体阈值是多少，需要看训练效果
def feature_price_corr(train):
    def spearman(frame, features):
        spr = pd.DataFrame()
        spr['feature'] = features
        spr['spearman'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]
        spr = spr.sort_values('spearman')
        plt.figure(figsize=(6, 0.25 * len(features)))
        sns.barplot(data=spr, y='feature', x='spearman', orient='h')
        plt.tight_layout(h_pad=None, w_pad=2.08)  # 可以设置图边距
        plt.show()

    features = quantitative + qual_encoded
    spearman(train, features)


#feature_price_corr(train)


# 数值特征之间、标称特征之间、数值和标称特征之间的相关系数
# 选取特征进行训练时，特征之间越独立（相关性越低）越好
def feature_feature_corr(train):
    plt.clf()
    plt.figure(1, figsize=(40, 40))
    corr = train[quantitative + ['SalePrice']].corr()
    cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)
    pic=sns.heatmap(corr, cmap=cmap, annot=True, fmt='.2f', annot_kws={'size':5})
    for text in pic.texts:
        if np.absolute(float(text.get_text())) <= 0.1:
            text.set_size(5)
            text.set_weight('bold')
            text.set_text('.1')
            #text.set_style('italic')
        elif np.absolute(float(text.get_text())) <= 0.2:
            text.set_size(5)
            text.set_weight('bold')
            text.set_text('.2')
        else:
            text.set_text(' ')
    plt.tight_layout(h_pad=2.08, w_pad=2.08)  # 可以设置图边距
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(fontsize=5)
    plt.show()

    plt.figure(2)
    corr = train[qual_encoded + ['SalePrice']].corr()
    cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, annot=True, fmt='.2f')
    plt.tight_layout(h_pad=2.08, w_pad=2.08)  # 可以设置图边距
    plt.xticks(rotation=90, fontsize=9)
    plt.show()

    plt.figure(3)
    corr = pd.DataFrame(np.zeros([len(quantitative) + 1, len(qual_encoded) + 1]), index=quantitative + ['SalePrice'],
                        columns=qual_encoded + ['SalePrice'])
    for q1 in quantitative + ['SalePrice']:
        for q2 in qual_encoded + ['SalePrice']:
            corr.loc[q1, q2] = train[q1].corr(train[q2])
    sns.heatmap(corr)
    plt.tight_layout(h_pad=2.08, w_pad=2.08)  # 可以设置图边距
    plt.xticks(rotation=90, fontsize=8)
    plt.show()


#feature_feature_corr(train)

def feature_feature_corr1(train):
    f, ax1 = plt.subplots(figsize=(10, 10))
    corr = train[quantitative + ['SalePrice']].corr()
    cmap = sns.cubehelix_palette(start=2, rot=3, gamma=0.8, as_cmap=True)
    pic=sns.heatmap(corr, cmap=cmap, annot=True, linewidths=.5, fmt='.1f', annot_kws={'size':7})

    for text in pic.texts:
        if np.absolute(float(text.get_text())) <= 0.1:
            text.set_size(7)
            #text.set_weight('bold')
            #text.set_text('.1')
            #text.set_style('italic')
        elif np.absolute(float(text.get_text())) <= 0.2:
            text.set_size(7)
            #text.set_weight('bold')
            #text.set_text('.2')
        else:
            #pass
            text.set_text(' ')
    plt.tight_layout(h_pad=2.08, w_pad=2.08)  # 可以设置图边距
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(fontsize=9)
    plt.show()

    f, ax2 = plt.subplots(figsize=(10, 10))
    corr = train[qual_encoded + ['SalePrice']].corr()
    cmap = sns.cubehelix_palette(start=2, rot=3, gamma=0.8, as_cmap=True)
    pic=sns.heatmap(corr, cmap=cmap, annot=True, linewidths=.5, fmt='.1f', annot_kws={'size':7})

    for text in pic.texts:
        if np.absolute(float(text.get_text())) <= 0.1:
            text.set_size(7)
            #text.set_weight('bold')
            #text.set_text('.1')
            #text.set_style('italic')
        elif np.absolute(float(text.get_text())) <= 0.2:
            text.set_size(7)
            #text.set_weight('bold')
            #text.set_text('.2')
        else:
            #pass
            text.set_text(' ')
    plt.tight_layout(h_pad=2.08, w_pad=2.08)  # 可以设置图边距
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(fontsize=9)
    plt.show()

    f, ax3 = plt.subplots(figsize=(10, 10))
    corr = pd.DataFrame(np.zeros([len(quantitative) + 1, len(qual_encoded) + 1]), index=quantitative + ['SalePrice'],
                        columns=qual_encoded + ['SalePrice'])
    for q1 in quantitative + ['SalePrice']:
        for q2 in qual_encoded + ['SalePrice']:
            corr.loc[q1, q2] = train[q1].corr(train[q2])
    cmap = sns.cubehelix_palette(start=2, rot=3, gamma=0.8, as_cmap=True)
    pic=sns.heatmap(corr, cmap=cmap, annot=True, linewidths=.5, fmt='.1f', annot_kws={'size':7})

    for text in pic.texts:
        if np.absolute(float(text.get_text())) <= 0.1:
            text.set_size(7)
            #text.set_weight('bold')
            #text.set_text('.1')
            #text.set_style('italic')
        elif np.absolute(float(text.get_text())) <= 0.2:
            text.set_size(7)
            #text.set_weight('bold')
            #text.set_text('.2')
        else:
            #pass
            text.set_text(' ')
    plt.tight_layout(h_pad=2.08, w_pad=2.08)  # 可以设置图边距
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(fontsize=9)
    plt.show()


#feature_feature_corr1(train)


def price_feature_pairplot(train):
    plt.clf()
    def pairplot(x, y, **kwargs):
        ax = plt.gca()
        ts = pd.DataFrame({'time': x, 'val': y})
        ts = ts.groupby('time').mean()
        ts.plot(ax=ax)
        plt.xticks(rotation=90)

    f = pd.melt(train, id_vars=['SalePrice'], value_vars=quantitative + qual_encoded)
    g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=4)
    g = g.map(pairplot, "value", "SalePrice")
    plt.tight_layout(h_pad=None, w_pad=None)  # 可以设置图边距
    plt.xticks(rotation=0, fontsize=9)
    plt.yticks(fontsize=9)
    plt.show()


#price_feature_pairplot(train)


def price_segments(train):
    plt.clf()
    features = quantitative

    standard = train[train['SalePrice'] < 200000]
    pricey = train[train['SalePrice'] >= 200000]

    diff = pd.DataFrame()
    diff['feature'] = features
    diff['difference'] = [
        (pricey[f].fillna(0.).mean() - standard[f].fillna(0.).mean()) / (standard[f].fillna(0.).mean()) for f in features]

    sns.barplot(data=diff, x='feature', y='difference')
    x = plt.xticks(rotation=90)
    plt.tight_layout(h_pad=1.08)
    plt.show()


#price_segments(train)


def clustering(train):
    plt.clf()
    features = quantitative + qual_encoded
    model = TSNE(n_components=2, random_state=0, perplexity=50)
    X = train[features].fillna(0.).values
    #tsne = model.fit_transform(X)  # 这里返回的是n行两列数据，因为是二维的，数据每一列代表一个维度

    std = StandardScaler()
    s = std.fit_transform(X)
    pca = PCA(n_components=30)
    pca.fit(s)
    pc = pca.transform(s)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(pc)

    tsne = model.fit_transform(pc)

    fr = pd.DataFrame({'tsne1': tsne[:, 0], 'tsne2': tsne[:, 1], 'cluster': kmeans.labels_})
    sns.lmplot(data=fr, x='tsne1', y='tsne2', hue='cluster', fit_reg=False)
    print(np.sum(pca.explained_variance_ratio_))
    plt.show()


#clustering(train)


def johnson_test(train):
    plt.clf()
    y = train['SalePrice'].values
    def johnson(y):
        gamma, eta, epsilon, lbda = st.johnsonsu.fit(y)
        yt = gamma + eta * np.arcsinh((y - epsilon) / lbda)
        return yt, gamma, eta, epsilon, lbda

    def johnson_inverse(y, gamma, eta, epsilon, lbda):
        return lbda * np.sinh((y - gamma) / eta) + epsilon

    yt, g, et, ep, l = johnson(y)
    yt2 = johnson_inverse(yt, g, et, ep, l)
    plt.figure(1)
    sns.distplot(yt)
    plt.figure(2)
    sns.distplot(yt2)
    plt.show()


#johnson_test(train)


# train and predict
def error(actual, predicted):
    actual = np.log(actual)
    predicted = np.log(predicted)
    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))

def log_transform(feature):
    train[feature] = np.log1p(train[feature].values)

def quadratic(feature):
    train[feature+'2'] = train[feature]**2


def train_func1():
    log_transform('GrLivArea')
    log_transform('1stFlrSF')
    log_transform('2ndFlrSF')
    log_transform('TotalBsmtSF')
    log_transform('LotArea')
    log_transform('LotFrontage')
    log_transform('KitchenAbvGr')
    log_transform('GarageArea')

    quadratic('OverallQual')
    quadratic('YearBuilt')
    quadratic('YearRemodAdd')
    quadratic('TotalBsmtSF')
    quadratic('2ndFlrSF')
    quadratic('Neighborhood_E')
    quadratic('RoofMatl_E')
    quadratic('GrLivArea')

    qdr = ['OverallQual2', 'YearBuilt2', 'YearRemodAdd2', 'TotalBsmtSF2',
           '2ndFlrSF2', 'Neighborhood_E2', 'RoofMatl_E2', 'GrLivArea2']

    train['HasBasement'] = train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    train['HasGarage'] = train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    train['Has2ndFloor'] = train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    train['HasMasVnr'] = train['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)
    train['HasWoodDeck'] = train['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
    train['HasPorch'] = train['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
    train['HasPool'] = train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    train['IsNew'] = train['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)

    boolean = ['HasBasement', 'HasGarage', 'Has2ndFloor', 'HasMasVnr', 'HasWoodDeck',
               'HasPorch', 'HasPool', 'IsNew']

    features = quantitative + qual_encoded + boolean + qdr
    lasso = linear_model.LassoLarsCV(max_iter=10000)
    X = train[features].fillna(0.).values
    Y = train['SalePrice'].values
    lasso.fit(X, np.log(Y))

    Ypred = np.exp(lasso.predict(X))
    print(error(Y, Ypred))


def train_func2():
    Y, X = patsy.dmatrices(
        "SalePrice ~ \
            GarageCars + \
            np.log1p(BsmtFinSF1) + \
            ScreenPorch + \
            Condition1_E + \
            Condition2_E + \
            WoodDeckSF + \
            np.log1p(LotArea) + \
            Foundation_E + \
            MSZoning_E + \
            MasVnrType_E + \
            HouseStyle_E + \
            Fireplaces + \
            CentralAir_E + \
            BsmtFullBath + \
            EnclosedPorch + \
            PavedDrive_E + \
            ExterQual_E + \
            bs(OverallCond, df=7, degree=1) + \
            bs(MSSubClass, df=7, degree=1) + \
            bs(LotArea, df=2, degree=1) + \
            bs(FullBath, df=3, degree=1) + \
            bs(HalfBath, df=2, degree=1) + \
            bs(BsmtFullBath, df=3, degree=1) + \
            bs(TotRmsAbvGrd, df=2, degree=1) + \
            bs(LandSlope_E, df=2, degree=1) + \
            bs(LotConfig_E, df=2, degree=1) + \
            bs(SaleCondition_E, df=3, degree=1) + \
            OverallQual + np.square(OverallQual) + \
            GrLivArea + np.square(GrLivArea) + \
            Q('1stFlrSF') + np.square(Q('1stFlrSF')) + \
            Q('2ndFlrSF') + np.square(Q('2ndFlrSF')) +  \
            TotalBsmtSF + np.square(TotalBsmtSF) +  \
            KitchenAbvGr + np.square(KitchenAbvGr) +  \
            YearBuilt + np.square(YearBuilt) + \
            Neighborhood_E + np.square(Neighborhood_E) + \
            Neighborhood_E:OverallQual + \
            MSSubClass:BldgType_E + \
            ExterQual_E:OverallQual + \
            PoolArea:PoolQC_E + \
            Fireplaces:FireplaceQu_E + \
            OverallQual:KitchenQual_E + \
            GarageQual_E:GarageCond + \
            GarageArea:GarageCars + \
            Q('1stFlrSF'):TotalBsmtSF + \
            TotRmsAbvGrd:GrLivArea",
        train.to_dict('list'))

    ridge = linear_model.RidgeCV(cv=10)
    ridge.fit(X, np.log(Y))
    Ypred = np.exp(ridge.predict(X))
    print(error(Y, Ypred))

#train_func1()
#train_func2()
def train_func3():
    #log_transform('GrLivArea')
    log_transform('1stFlrSF')
    #log_transform('2ndFlrSF')
    #log_transform('TotalBsmtSF')
    #log_transform('LotArea')
    log_transform('LotFrontage')
    #log_transform('KitchenAbvGr')
    #log_transform('GarageArea')

    quadratic('OverallQual')
    quadratic('YearBuilt')
    quadratic('YearRemodAdd')
    quadratic('TotalBsmtSF')
    quadratic('2ndFlrSF')
    quadratic('Neighborhood_E')
    quadratic('RoofMatl_E')
    quadratic('GrLivArea')

    qdr = ['OverallQual2', 'YearBuilt2', 'YearRemodAdd2', 'TotalBsmtSF2',
           '2ndFlrSF2', 'Neighborhood_E2', 'RoofMatl_E2', 'GrLivArea2']

    #qdr = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF',
    #       '2ndFlrSF', 'Neighborhood_E', 'RoofMatl_E', 'GrLivArea']

    train['HasBasement'] = train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    train['HasGarage'] = train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    train['Has2ndFloor'] = train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    train['HasMasVnr'] = train['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)
    train['HasWoodDeck'] = train['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
    train['HasPorch'] = train['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
    train['HasPool'] = train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    train['IsNew'] = train['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)

    boolean = ['HasBasement', 'HasGarage', 'Has2ndFloor', 'HasMasVnr', 'HasWoodDeck',
               'HasPorch', 'HasPool', 'IsNew']

    features = quantitative + qual_encoded + boolean + qdr
    lasso = linear_model.LassoLarsCV(max_iter=10000)
    X = train[features].fillna(0.).values
    Y = train['SalePrice'].values
    lasso.fit(X, np.log(Y))

    Ypred = np.exp(lasso.predict(X))
    print(error(Y, Ypred))


    # predict

    lasso.predict()
#train_func3()


def train_predict():
    def log_transform1(feature, data):
        data[feature] = np.log1p(data[feature].values)

    def quadratic1(feature, data):
        data[feature+'2'] = data[feature]**2

    def data_pre_process(data):
        log_transform1('1stFlrSF', data)
        log_transform1('LotFrontage', data)
        #log_transform1('GrLivArea', data)
        #log_transform1('2ndFlrSF', data)
        #log_transform1('TotalBsmtSF', data)
        #log_transform1('LotArea', data)
        #log_transform1('KitchenAbvGr', data)
        #log_transform1('GarageArea', data)

        quantitative1 = ['1stFlrSF', 'LotFrontage', 'GrLivArea', '2ndFlrSF', 'TotalBsmtSF',
                         'LotArea', 'KitchenAbvGr', 'GarageArea']

        importance_scalar = ['Neighborhood_E', 'ExterQual_E', 'BsmtQual_E', 'KitchenQual_E',
                             'GarageFinish_E', 'GarageType_E', 'Foundation_E', 'FireplaceQu_E']
        importance_number = ['OverallQual', 'YearBuilt', 'GarageCars', 'FullBath', 'GarageYrBlt',
                             'YearRemodAdd', 'TotRmsAbvGrd', 'Fireplaces']

        quadratic1('OverallQual', data)
        quadratic1('YearBuilt', data)
        quadratic1('YearRemodAdd', data)
        quadratic1('TotalBsmtSF', data)
        quadratic1('2ndFlrSF', data)
        quadratic1('Neighborhood_E', data)
        quadratic1('RoofMatl_E', data)
        quadratic1('GrLivArea', data)

        qdr = ['OverallQual2', 'YearBuilt2', 'YearRemodAdd2', 'TotalBsmtSF2',
               '2ndFlrSF2', 'Neighborhood_E2', 'RoofMatl_E2', 'GrLivArea2']

        data['HasBasement'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
        data['HasGarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
        data['Has2ndFloor'] = data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
        data['HasMasVnr'] = data['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)
        data['HasWoodDeck'] = data['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
        data['HasPorch'] = data['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
        data['HasPool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
        data['IsNew'] = data['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)

        boolean = ['HasBasement', 'HasGarage', 'Has2ndFloor', 'HasMasVnr', 'HasWoodDeck',
                   'HasPorch', 'HasPool', 'IsNew']

        qual_encoded_bak = list(qual_encoded)
        quantitative_bak = list(quantitative)

        quantitative_bak.remove('OverallCond')
        quantitative_bak.remove('LowQualFinSF')
        quantitative_bak.remove('MiscVal')
        quantitative_bak.remove('BsmtFinSF2')
        quantitative_bak.remove('YrSold')
        quantitative_bak.remove('MSSubClass')
        quantitative_bak.remove('PoolArea')
        quantitative_bak.remove('MoSold')
        quantitative_bak.remove('EnclosedPorch')
        quantitative_bak.remove('KitchenAbvGr')

        # pair show
        quantitative_bak.remove('BsmtFullBath')
        quantitative_bak.remove('HalfBath')
        quantitative_bak.remove('BedroomAbvGr')
        quantitative_bak.remove('WoodDeckSF')
        quantitative_bak.remove('OpenPorchSF')
        quantitative_bak.remove('3SsnPorch')
        quantitative_bak.remove('ScreenPorch')

        quantitative_bak.remove('GarageCars')
        quantitative_bak.remove('GarageYrBlt')

        qual_encoded_bak.remove('Utilities_E')
        qual_encoded_bak.remove('Street_E')
        qual_encoded_bak.remove('LandSlope_E')
        qual_encoded_bak.remove('PoolQC_E')
        qual_encoded_bak.remove('LotConfig_E')
        qual_encoded_bak.remove('GarageFinish_E')
        qual_encoded_bak.remove('GarageType_E')

        #features = quantitative_bak + qual_encoded_bak + boolean + qdr
        features = quantitative + qual_encoded + boolean + qdr
        #features = quantitative1 + importance_scalar +importance_number + boolean + qdr

        return features

    def train_model(features):
        lasso = linear_model.LassoLarsCV(max_iter=10000)
        X = train[features].fillna(0.).values
        Y = train['SalePrice'].values
        lasso.fit(X, np.log(Y))

        Ypred = np.exp(lasso.predict(X))
        print(error(Y, Ypred))
        return lasso

    def predict_data(model):
        scalar_feature_fill_missing(test)
        scalar_encode_by_list(encode_map_list, test)
        features = data_pre_process(test)
        x_need_pred = test[features].fillna(0.).values
        y_pred = model.predict(x_need_pred)
        return y_pred

    features = data_pre_process(train)
    model = train_model(features)

    y_pred = predict_data(model)
    real_y = np.exp(y_pred)
    pd.DataFrame([[i, real_y[i - 1461]] for i in range(1461, 1461+len(real_y))], ).to_csv(cur_dir + "/price_predict.csv", header=['Id', 'SalePrice'], index=False)

train_predict()



