'''Help to create a data explorer file'''
from recommender import *
import pandas as pd
import streamlit as st
import unicodedata
import base64
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import urllib
from PIL import Image
import plotly.express as px

px.defaults.width = 800
px.defaults.height = 400
px.defaults.template = 'ggplot2'

# import unidecodedata
sns.set(style="darkgrid")

@st.cache
def get_table_download_link(df):
    """
    Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href



def main():

    st.title('Leads Recommender B2B')
    
    file_path = 'C:/Users/simon/Documents/codenation_projeto_final/' 
    b2bimage = Image.open(file_path+'app/b2b.png')
    st.sidebar.image(b2bimage,  use_column_width=True, format = 'PNG')

    market = pd.read_csv(file_path+'data/estaticos_market.csv', index_col=0)
    market.dropna(subset = ['setor',  'nm_divisao', 'nm_segmento'], axis = 0, inplace = True)
    market.reset_index(inplace=True, drop=True)

    numeric_features, categorical_features, _ = categorical_types(market)

    col_list = ['id', 'fl_matriz',  'de_natureza_juridica',  'sg_uf',  'natureza_juridica_macro','de_ramo',
            'setor','idade_emp_cat', 'fl_me', 'fl_sa', 'fl_epp', 'fl_mei', 'fl_ltda', 'fl_st_especial', 'fl_rm', 
            'nm_divisao', 'nm_segmento', 'fl_spa', 'fl_antt', 'fl_veiculo', 'vl_total_veiculos_pesados_grupo', 
            'vl_total_veiculos_leves_grupo', 'de_saude_tributaria', 'de_saude_rescencia', 'de_nivel_atividade', 
            'fl_simples_irregular', 'empsetorcensitariofaixarendapopulacao', 'nm_meso_regiao', 'nm_micro_regiao', 
            'fl_passivel_iss', 'qt_socios_pf', 'qt_socios_pj', 'idade_media_socios', 'qt_socios_st_regular',
            'de_faixa_faturamento_estimado', 'de_faixa_faturamento_estimado_grupo', 'qt_filiais']
    
    _, _, categ_bool = categorical_types(market[col_list])
    
    df = preprocess(market[col_list].copy())

    file_type = st.sidebar.radio('Select the portfolio', ('Portfolio 1','Portfolio 2','Portfolio 3'))

    if file_type == 'Portfolio 1':
        select = 1
    if file_type == 'Portfolio 2':
        select = 2
    if file_type == 'Portfolio 3':
        select = 3
        
    portfolio = pd.read_csv(file_path + f'data/estaticos_portfolio{select}.csv', index_col=0)
    portfolio = portfolio[['id']]


    n_items = len(portfolio)
    neighbors = int(0.5*len(portfolio))
    user_companies = pd.merge(portfolio, df, on='id', how='inner', left_index=True)
    leads_index = recommender_leads(user_companies, df, categ_bool, select, 'func3', neighbors, max_leads = None)

    
    portfolio_companies = pd.merge(portfolio, market, on='id', how='inner', left_index=True)
    portfolio_leads = market.iloc[leads_index[:2*n_items],:]
    
    nleads = st.slider('How many leads do you want to display?', 10,len(leads_index))

    if st.checkbox('Show leads ID'):
        st.table(market.iloc[leads_index[:nleads],0])
    
    #Sidebar Menu
    options = ["WordCloud", "Comparison", 'Profiling', 'Location x Market division', 'Market Revenue']
             
    menu = st.sidebar.selectbox("Menu options", options)

    if menu == ('Location x Market division'):
        
        cols = [['nm_divisao'],['sg_uf', 'nm_meso_regiao']]

        na = True
        ax = 1
        normalize=True

        crosstable_companies =heatmap(portfolio_companies, cols, na,ax, normalize)
        crosstable_leads = heatmap(portfolio_leads, cols, na, ax, normalize)
        # st.table(crosstable_companies)
        crosstable_companies.columns = crosstable_companies.columns.map(' | '.join).str.strip('|')
        crosstable_leads.columns = crosstable_leads.columns.map(' | '.join).str.strip('|')
        
        st.header("Customers")
        x = list(crosstable_companies.columns)
        y = list(crosstable_companies.index)
        fig_companies = px.imshow(crosstable_companies, x =x, y=y, color_continuous_scale=px.colors.sequential.Greys)
        fig_companies.update_xaxes(side="top")
        fig_companies.update_layout(autosize=True, width=800, height=800)
        st.plotly_chart(fig_companies)

        st.header("Leads")
        x = list(crosstable_leads.columns)
        y = list(crosstable_leads.index)
        fig_leads = px.imshow(crosstable_leads, x =x, y=y, color_continuous_scale=px.colors.sequential.Greys)
        fig_leads.update_xaxes(side="top")
        fig_leads.update_layout( autosize=True, width=800, height=800)
        st.plotly_chart(fig_leads)

    if (menu =="Comparison"):
        st.header("Comparison of Customer x Leads")
        values = ['sg_uf', 'natureza_juridica_macro', 'setor', 'de_faixa_faturamento_estimado', 'idade_emp_cat','de_ramo']
        
        for cols in values:
            # cols = values[0]

            aux_df1 = pd.crosstab(
                    index = portfolio_companies[cols], 
                    columns='Total',
                    dropna=True,
                    normalize = True)
            aux_df1['type'] = 'customer'
            aux_df2 = pd.crosstab(
                    index = portfolio_leads[cols], 
                    columns='Total',
                    dropna=True,
                    normalize = True)#.loc[aux_df1.index,:]
            aux_df2['type'] = 'leads'
            
            indexes = list(set(aux_df1.index) & set(aux_df2.index))

            conc = pd.concat([aux_df1.loc[indexes],aux_df2.loc[indexes]], axis = 0)
            # st.table(conc)

            fig = px.line_polar(conc, r='Total', theta=list(conc.index), color = 'type', line_close=True,\
                                color_discrete_sequence=px.colors.qualitative.Vivid,)

            fig.update_traces(fill='toself')
            fig.update_layout(title_text=f'Comparison Customers x Leads by {cols}')
            st.plotly_chart(fig)


    if (menu =="Profiling"):
        st.header("Profiling Customers x Leads")

        values = ['idade_empresa_anos', 'empsetorcensitariofaixarendapopulacao',\
                 'qt_socios', 'qt_filiais','vl_faturamento_estimado_aux', 'idade_media_socios', \
                 'qt_funcionarios', 'qt_funcionarios_grupo', 'tx_crescimento_24meses']
        filter_data = st.sidebar.selectbox('Select numeric variable to visualize as color', values, key='plot_map')
            
        cols = ['setor', 'de_ramo', 'natureza_juridica_macro']
    
        fig_companies = px.parallel_categories(portfolio_companies, 
                            dimensions = cols,
                            color=filter_data, 
                            color_continuous_scale=px.colors.sequential.Inferno)
        fig_companies.update_layout(title_text=f'Customers',  autosize=False, width=800, height=600)

        fig_leads =  px.parallel_categories(portfolio_leads, 
                            dimensions = cols,
                            color=filter_data, 
                            color_continuous_scale=px.colors.sequential.Inferno)
        fig_leads.update_layout(title_text=f'Leads', autosize=False, width=800, height=600)


        st.plotly_chart(fig_companies)
        st.plotly_chart(fig_leads)
        
    
    if menu == 'Market Revenue':

        portfolio_companies['market'] = 'market'
        fig_companies = px.treemap( portfolio_companies, path =  ['market', 'setor', 'nm_segmento','nm_divisao'], values = 'qt_funcionarios',
        color = 'vl_faturamento_estimado_aux', color_continuous_scale='RdBu')
        fig_companies.update_layout(title_text=f'Customers',  autosize=False, width=800, height=400)
        st.plotly_chart(fig_companies,use_container_width=True)

        portfolio_leads['market'] = 'market'
        fig_leads = px.treemap( portfolio_leads, path =  ['market', 'setor', 'nm_segmento','nm_divisao'], values = 'qt_funcionarios',
        color = 'vl_faturamento_estimado_aux', color_continuous_scale='RdBu')
        fig_leads.update_layout(title_text=f'Leads', autosize=False, width=800, height=400)
        st.plotly_chart(fig_leads,use_container_width=True)

    if (menu == "WordCloud"):  
       
        #Definindo a lista de stopwords
        stopwords= set(STOPWORDS)
        #Adicionando a lista stopwords em português
        http = 'https://gist.githubusercontent.com/alopes/5358189/raw/2107d809cca6b83ce3d8e04dbd9463283025284f/stopwords.txt'
        file = urllib.request.urlopen(http)
        stopwords_pt = []
        
        for line in file:
            for word in line.decode("utf-8").strip().split():
                stopwords_pt.append(word)
            
        set_stopwords = stopwords.union(set(stopwords_pt))
        #st.write(new_stopwords)
        wordcloud_column = 'soup'
        comment_words = ''     
        
        def create_soup(x):
            return x['sg_uf'] + ' ' + x['de_natureza_juridica'] + ' ' + x['setor'] + ' '+ x['de_ramo'] + ' ' + x['nm_divisao']\
                + ' ' + x['nm_segmento']  
        
        df = portfolio_companies.copy()
        df.fillna("", inplace=True)
        df['soup'] = portfolio_companies.apply(create_soup, axis=1)

        #st.table(df.soup.head())

        for val in df[wordcloud_column].dropna(): 
            val = str(val)
            tokens = val.split() 
            
            # Converts each token into lowercase 
            for i in range(len(tokens)): 
                tokens[i] = tokens[i].lower() 
            
            comment_words += " ".join(tokens)+" "
        
        wordcloud = WordCloud(width = 1200, height = 600,
                        stopwords = set_stopwords,
                        background_color ='white', 
                        min_font_size = 12,
                        normalize_plurals= True).generate(comment_words) 
        # text = " ".join(i for i in df[wordcloud_column])
        # wordcloud = WordCloud().generate(text)
        plt.imshow(wordcloud, interpolation='hamming')
        plt.axis("off")
        st.pyplot()

 
    st.sidebar.warning('All the leads visualizations considers only `2*len(portfolio)`')
    st.sidebar.title('About')
    st.sidebar.info('This app is a data explorer tool available on [ExploreDataset](https://exploredataset.herokuapp.com/) \n \
    It is mantained by [Simone](https://www.linkedin.com/in/simonezambonim/). Check this code on [here](https://github.com/simonezambonim/explore_dataset)')


if __name__ == '__main__':
    main()

