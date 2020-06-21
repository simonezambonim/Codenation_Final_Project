import pandas as pd
import gower
import numpy as np
from sklearn.model_selection import train_test_split
import pickle


file_path = 'C:/Users/simon/Documents/codenation_projeto_final/app/'

def preprocess(df):
    
    # Fill numeric values with the mean\median
    df['idade_media_socios'].fillna(int(df['idade_media_socios'].mean()), inplace=True)
    df['empsetorcensitariofaixarendapopulacao'].fillna(df['empsetorcensitariofaixarendapopulacao'].median(), inplace=True)
    df['empsetorcensitariofaixarendapopulacao'] = np.log(df['empsetorcensitariofaixarendapopulacao'])

    # Fill these with bussiness info
    df['qt_socios_pf'].fillna(1, inplace=True)
    df['qt_socios_pj'].fillna(0, inplace=True)
    df['qt_socios_st_regular'].fillna(df['qt_socios_pf'], inplace=True)

    # Fill nominal/ordinal categorical data with 'SEM INFORMACAO'
    to_fill = ['de_faixa_faturamento_estimado','de_faixa_faturamento_estimado_grupo',
    'de_nivel_atividade', 'de_saude_rescencia','nm_meso_regiao','nm_micro_regiao']

    df['de_saude_tributaria'].fillna('CINZA')

    for x in to_fill:
        df[x].fillna('SEM INFORMACAO', inplace=True)

    #Encode the ordinal variables, following a ordinal ogic
    dic_faturamento_estimado = {
        'SEM INFORMACAO' : 0,
        'ATE R$ 81.000,00' : 1,
        'DE R$ 81.000,01 A R$ 360.000,00' : 2,
        'DE R$ 360.000,01 A R$ 1.500.000,00':3,
        'DE R$ 1.500.000,01 A R$ 4.800.000,00':4,
        'DE R$ 4.800.000,01 A R$ 10.000.000,00':5,
        'DE R$ 10.000.000,01 A R$ 30.000.000,00':6, 
        'DE R$ 30.000.000,01 A R$ 100.000.000,00':7,
        'DE R$ 100.000.000,01 A R$ 300.000.000,00':8,
        'DE R$ 300.000.000,01 A R$ 500.000.000,00':9,
        'DE R$ 500.000.000,01 A 1 BILHAO DE REAIS':10,
        'ACIMA DE 1 BILHAO DE REAIS':11}

    dic_idade = {
        '<= 1': 1,
        '1 a 5': 2,
        '5 a 10': 3,
        '10 a 15':4,
        '15 a 20':5,
        '> 20' :6}

    dic_de_nivel_atividade = {
        'SEM INFORMACAO': 0,
        'MUITO BAIXA': 1,
        'BAIXA':2,
        'MEDIA':3,
        'ALTA': 4
        }    

    dic_de_saude_rescencia = {'SEM INFORMACAO': 0,
    'ATE 3 MESES' : 1,
    'ATE 6 MESES' :2,
    'ATE 1 ANO':3,
    'ACIMA DE 1 ANO':4}

    dic_de_saude_tributaria = {
    'CINZA': 0,
    'VERMELHO':1,
    'LARANJA': 2,
    'AMARELO':3,
    'AZUL':4,
    'VERDE':5}

    # Map 
    df['de_faixa_faturamento_estimado'] = df['de_faixa_faturamento_estimado'].map(dic_faturamento_estimado)
    df['de_faixa_faturamento_estimado_grupo'] = df['de_faixa_faturamento_estimado_grupo'].map(dic_faturamento_estimado)
    df['idade_emp_cat'] = df['idade_emp_cat'].map(dic_idade)
    df['de_nivel_atividade'] = df['de_nivel_atividade'].map(dic_de_nivel_atividade)
    df['de_saude_rescencia'] = df['de_saude_rescencia'].map(dic_de_saude_rescencia)
    df['de_saude_tributaria'] = df['de_saude_tributaria'].map(dic_de_saude_tributaria)

    # Encode the bool features
    dic_bool = {
        'SIM': 1,
        'NAO': 0}

    df['fl_rm'] = df['fl_rm'].map(dic_bool)

    bool_features = [col for col in df.columns if col.startswith('fl_')]

    # Treat them as integers
    for feat in bool_features:    
        df[feat] = df[feat].astype(int)

    # Encode the remaining categorical variables 
    categorical_features = [
        'natureza_juridica_macro',
        'de_natureza_juridica',
        'de_ramo',
        'setor',
        'nm_divisao',
        'nm_segmento',
        'nm_meso_regiao',
        'nm_micro_regiao',
        'sg_uf']

    # keep track of the encoding in the map_categories variable
    map_categories = dict()
    for feat in categorical_features:
        df[feat] = df[feat].astype('category')
        encode_categ = df[feat].cat.codes
        map_categories[feat] = dict(zip(encode_categ, df[feat]))
        df[feat] = encode_categ

    return df

def dropna_cols(df,info,max_na): 
    '''
    Function to drop columns above threshold max_na
    '''
    return df.drop(info[info['nan%']  > max_na].index, axis=1).copy()

def categorical_types(df):
    '''
    Our model requires to know where are the positions of each categorical variable
    Here we extract this info!

    returns
    : all_features - complete set of columns
    : categ_features -  boolean and object features 
    : num_features - numeric features
    : categ_bool - the whole index of columns stating True for categorical variables
    '''
    
    all_features = df.columns.to_list() 
    num_features = df.select_dtypes(['int64','float']).columns.to_list()
    categ_features = df.select_dtypes(['bool','object']).columns.to_list()
    categ_func = lambda x: True if x in categ_features else False
    categ_bool = [categ_func(x) for x in all_features]
        
    return num_features, categ_features, categ_bool


def get_user_companies(file_path,file):
    return pd.read_csv(file_path + file, index_col=0)


def gower_metric(user_companies, market, categ_bool, select):
    '''
    Calculates pairwise distance between the market database and the user portfolio 
    using the Gower Coefficient
    '''
    #gower.gower_matrix(market.iloc[:,1:], user_companies.iloc[:,1:], cat_features=categ_bool[1:])
    return pickle.load(open(f'gower_{select}.pkl', 'rb'))

def recommender_leads(user_companies, market, categ_bool, select, function='func3', neighbors = 50, max_leads = None):
    '''
    Rocommend leads based on similarities
    Uses Gower Coefficiente as metric

    param market :: market dataframe 
    param user_companies: dataframe with companies in the user porfolio
    param categ_bool: list of boolean required to indicate if a feature is categorical or not
    param k : numero de recomendações a serem geradas
    returns: index of recommended leads

    # Hyperparameters that define precision
    max_leads :  list size (the larger, more likely one of the portfolio companies to be in the recommended list)
    neighbors : number of nearby companies to be considered ''in the rating''

    '''
    #We are not calculating the distance matrix, due to the time required.. we are loading it 
    gower_dist =  gower_metric(user_companies, market, categ_bool, select)
    #gower_dist = pickle.load(open(file_path+f'gower_{len(user_companies)}.pkl', 'rb'))

    ind = []
    dist = []
    for i in range(len((user_companies.id))):
      ind_aux = np.argpartition(gower_dist[:,i], neighbors, axis=-1)[:neighbors]

      #remove user items
      valid_mask = np.isin(ind_aux , user_companies.index.values, assume_unique=True, invert=True)

      dist_aux = gower_dist[ind_aux[valid_mask],i]
      ind.append(ind_aux[valid_mask])
      dist.append(dist_aux)
  
    concat_ind = np.concatenate(ind)
    concat_dist = np.concatenate(dist)  

    ## Check for reapeted recomendation and count
    unique, counts = np.unique(concat_ind, return_counts=True) 

    # Score candidates
    relevance = [] 


    for n_times,i in sorted(zip(counts, unique), reverse=True):
      if function == 'func0':
        index_ = np.where(concat_ind == i)
        rating = np.mean(concat_dist[index_[0]])
      if function == 'func1':
        rating = 1/n_times
      elif function == 'func2':
        rating =  np.linalg.norm(gower_dist[i,:]) 
      elif function == 'func3':
        rating =  ((1/n_times)**2)*np.linalg.norm(gower_dist[i,:])  
      relevance.append([i, rating])

    ## Let's sort the best candidates and get the k best candidates
    best_candidates = sorted(relevance, key=lambda x: x[1])

    # Retrieving the indexes
    top_index = []
    for idx,val in best_candidates: 
        top_index.append(idx)

    if max_leads == None or max_leads > len(top_index):
      return top_index
    else:
      return top_index[:max_leads]


def contigency_table(df, cols, na=False, ax=0, normalize=False):
    '''
    cols :: this is a list of lists 
            it must have length of 2
            col[0] contains the values to group by in the index
            col[1] contains the values to group by in the columns
    na:: Do not include columns whose entries are all NaN
    ax:: Axis to consider the bar styling % 
        if 0 ->(rows) 
        if 1 -> (columns) 
    normalize:: Normalize by dividing all values by the sum of values
                If passed ‘all’ or True, will normalize over all values.
                If passed ‘index’ will normalize over each row.
                If passed ‘columns’ will normalize over each column.
                If margins is True, will also normalize margin values.  

    '''

    if len(cols)<=2:
        if len(cols) == 1:
            crosstable = pd.crosstab(
                index = df[cols[0][0]], 
                columns="Total",
                dropna=na,
                normalize = normalize)
        elif len(cols) == 2: 
            if len( cols[0]) == 1 and len( cols[1]) == 1:
                crosstable = pd.crosstab(
                    index = df[cols[0][0]], 
                    columns = df[cols[1][0]],  
                    rownames= cols[0],
                    colnames= cols[1],
                    dropna=na,
                    margins=True, 
                    margins_name="Total",
                    normalize = normalize)
            elif len( cols[0]) == 1 and len( cols[1]) == 2:
                crosstable = pd.crosstab(
                    index = df[cols[0][0]], 
                    columns =[df[cols[1][0]], df[cols[1][1]]], 
                    rownames= cols[0],
                    colnames= cols[1],
                    dropna=na,
                    margins=True, 
                    margins_name="Total",
                    normalize = normalize)
            elif len( cols[0]) == 2 and len( cols[1]) == 1:
                crosstable = pd.crosstab(
                    index =[df[cols[0][0]], df[cols[0][1]]], 
                    columns = df[cols[1][0]], 
                    rownames= cols[0],
                    colnames= cols[1],
                    dropna=na,
                    margins=True, 
                    margins_name="Total",
                    normalize = normalize)    
            elif len( cols[0]) == 2 and len( cols[1]) == 2:
                crosstable = pd.crosstab(
                    index =[df[cols[0][0]],df[cols[0][1]]],
                    columns =[ df[cols[1][0]], df[cols[1][1]]], 
                    rownames=cols[0],
                    colnames= cols[1],
                    dropna=na,
                    margins=True, 
                    margins_name="Total",
                    normalize = normalize)   
       
        return crosstable.style.bar(color='#3d66af', axis = ax)#.background_gradient( axis=None)


def heatmap(df, cols,na=False,ax=0,normalize=False):
    '''
    cols :: this is a list of lists 
            it must have length of 2
            col[0] contains the values to group by in the index
            col[1] contains the values to group by in the columns
    na:: Do not include columns whose entries are all NaN
    ax:: Axis to consider the bar styling % 
        if 0 ->(rows) 
        if 1 -> (columns) 
    normalize:: Normalize by dividing all values by the sum of values
                If passed ‘all’ or True, will normalize over all values.
                If passed ‘index’ will normalize over each row.
                If passed ‘columns’ will normalize over each column.
                If margins is True, will also normalize margin values.  

    '''

    if len(cols)<=2:
        if len(cols) == 1:
            crosstable = pd.crosstab(
                index = df[cols[0][0]], 
                columns="Total",
                dropna=na,
                normalize = normalize)
        elif len(cols) == 2: 
            if len( cols[0]) == 1 and len( cols[1]) == 1:
                crosstable = pd.crosstab(
                    index = df[cols[0][0]], 
                    columns = df[cols[1][0]],  
                    rownames= cols[0],
                    colnames= cols[1],
                    dropna=na,
                    normalize = normalize)
            elif len( cols[0]) == 1 and len( cols[1]) == 2:
                crosstable = pd.crosstab(
                    index = df[cols[0][0]], 
                    columns =[df[cols[1][0]], df[cols[1][1]]], 
                    rownames= cols[0],
                    colnames= cols[1],
                    dropna=na,
                    normalize = normalize)
            elif len(cols[0]) == 2 and len( cols[1]) == 1:
                crosstable = pd.crosstab(
                    index =[df[cols[0][0]], df[cols[0][1]]], 
                    columns = df[cols[1][0]], 
                    rownames= cols[0],
                    colnames= cols[1],
                    dropna=na,
                    normalize = normalize)    
            elif len( cols[0]) == 2 and len( cols[1]) == 2:
                crosstable = pd.crosstab(
                    index =[df[cols[0][0]],df[cols[0][1]]],
                    columns =[ df[cols[1][0]], df[cols[1][1]]], 
                    rownames=cols[0],
                    colnames= cols[1],
                    dropna=na,
                    normalize = normalize)   
        

        return crosstable#.apply(lambda x : x / x.sum(), axis=ax)

