from cmath import log
from genericpath import exists
from json import tool
from unicodedata import name
import kasempy as kp
import json
from sklearn.neighbors import KernelDensity
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import os 

#Parsing JSON
f = open('tool_wear_script.json', 'r')
param = json.load(f)
data = param['data']
logs = param['plateform']
f.close()





onglets = {}
cols = {}
for j in data:
    tool_num = j['name']
    bandw = j['bandwidths']
    var_name = j['features_var']
    onglets[tool_num] = []
    


    #Calling Kasempy API

    dsb = kp.DatasetBuilder(logs['host'], logs['user'],  logs['password'])
    agents = dsb.api.agents()
    agent = agents.loc[agents['name'] == logs['agent']].index
    dsb.api.set_agent(agent[0])

    dsb.set_block(logs['agent'])
    dsb.add('program','OP390_Program')
    dsb.build() 
    var = dsb.api.variables()
    id = var.loc[var['name'] == var_name].index
    
    api = kp.RestAPI(logs['host'], logs['user'], logs['password'])
    api.set_agent(agent[0])
    tab_values = api.variable_get_history_data(id[0])
    #Coverting values into df
    col = j['features_label']
    df = pd.DataFrame(columns= col)
    for i in tab_values.index:    
        new_row = pd.Series(tab_values.iloc[i]['value'], index=col)
        df = df.append(new_row, ignore_index=True)
    df = df.astype('float64')
    df['ts'] = tab_values['date'].values.astype(np.int64) // 10 ** 9
    df = df.dropna()

    kdes = {}
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, tool_num)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    #Bandwidth selection
    if (bandw == None):
        bandw = []
        grid = GridSearchCV(KernelDensity(kernel = 'gaussian'),{'bandwidth': np.linspace(0.05, 0.5, 20)}, cv = 5)
        grid.fit(df[col])
        temp = grid.best_estimator_
        bandw.append(temp.bandwidth)        
   
    #Storing kde(s)
    for i in bandw:
        path = os.path.join(current_directory + '/' + tool_num + '/', str(i))
        if not os.path.exists(path):
            os.mkdir(path)
        temp = KernelDensity(kernel='gaussian', bandwidth=i).fit(df[col])
        #Saving each model
        path = os.path.join(path,'kde_' + str(i))
        with open(path, 'wb') as f:
            pickle.dump(temp,f)
        df[str(temp.bandwidth)] = temp.score_samples(df[col])
        kdes[str(temp.bandwidth)] = temp
        onglets[tool_num] += [i]
            
    #Graphs
    for kde in kdes:
        path = os.path.join(current_directory + '/' + tool_num + '/')
        #Pairplot plot
        plt.figure(figsize=(12,6))
        pd.plotting.scatter_matrix(df[col], figsize=(12,6), c=df[kde], hist_kwds = {'bins': 100}, grid= True)
        plt.savefig(path + str(kde) +'/pairplot_'+ str(kde)+'.png')


        for c in col:
            path = current_directory + '/' + tool_num + '/' + str(kde)
            c_lower_bound = df[kde].min()
            c_upper_bound = df[kde].max()

            #Mean plot
            plt.figure(figsize=(12,6))
            sns.kdeplot(df[c], color='b', shade=True, bw_method = float(kde))
            plt.twinx().hist(x = df[c], bins=100, alpha = .4)
            plt.title(tool_num+ '_' + str(kde))
            plt.grid()
            plt.savefig(path +'/kde_plot_dens_' + c + '_'+ kde+'.png')
        

            plt.figure(figsize=(12,6))
            plt.scatter(df.index, df[c], c = df[kde], alpha = 0.8, vmin = c_lower_bound, vmax = c_upper_bound)
            plt.ylabel(c)
            cba = plt.colorbar()
            cba.set_label('log_prob')
            plt.title(tool_num+ '_' + str(kde))
            plt.grid()
            plt.savefig(path +'/kde_plot_'+ c + '_'+ kde +'.png')
            
            plt.close('all')
            cols[tool_num] = col


    for o in onglets:
        col = cols[o]
        for i in onglets[o]:
            #HTML Generation
            path = './' + o + '/' + str(i)
            f = open ('./' + o + '/' + str(i) + '/kde_'+ str(i) + ".html", 'w')
            text = '''<!DOCTYPE html>
                    <html lang="fr">
                    <head>
                        <meta charset="UTF-8">
                        <title>InterQ_KDE_visuals</title>
                    </head>
                    <body>
                    <nav>'''
            f.write(text)
            f.write('<h2>Features Distribution</h2><img src= ./pairplot_' + str(i) + '.png/>')
            for c in col:
                text='<h2>'+ c +'</h2><img src= ./kde_plot_dens_' + c + '_' + str(i) + '.png/><img src=./kde_plot_' + c + '_' + str(i) + '.png/>'
                f.write(text)
            f.write('<div class="l-navbar" id="nav-bar"><nav class="nav"><div class="l-navbar" id="nav-bar"><nav class="nav"><div>')
            
            for w in onglets:
                f.write('<span class="nav_name"></span> </a> <a href="#" class="nav_link"><span class="nav_name" id="overview"' + ')><strong>' + w + '</strong></span> </a>')
                if w == o:
                    for x in onglets[w]:
                        text= '<span class="nav_name"></span> </a> <a href="#" class="nav_link"><span class="nav_name" id="overview" onclick="window.location.href='+ "'" + '../' + str(x) + '/kde_' + str(x) + '.html' + "'" + '";>' + str(round(x, 3)) + '</span> </a>'
                        f.write(text)
                else:
                    for x in onglets[w]:
                        text= '<span class="nav_name"></span> </a> <a href="#" class="nav_link"><span class="nav_name" id="overview" onclick="window.location.href='+ "'../../"  + w + '/' + str(x) + '/kde_' + str(x) + '.html' + "'" + '";>' + str(round(x, 3)) + '</span> </a>'
                        f.write(text)
            f.write('</div></nav></div>')
            text = '''<style>
            @import url("https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap");

        :root {
    --header-height: 3rem;
    --first-color: #4723D9;
    --first-color-light: #AFA5D9;
    --white-color: #F7F6FB;
    --body-font: 'Nunito', sans-serif;
    --normal-font-size: 1rem;
    --z-fixed: 100
    }

    *,
    ::before,
    ::after {
    box-sizing: border-box
    }

    body {
    position: relative;
    margin: var(--header-height) 0 0 0;
    padding: 0 1rem;
    font-family: var(--body-font);
    font-size: var(--normal-font-size);
    transition: .5s
    }

    a {
    text-decoration: none
    }

    .header {
    width: 100%;
    height: var(--header-height);
    position: fixed;
    top: 0;
    left: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 1rem;
    background-color: var(--white-color);
    z-index: var(--z-fixed);
    transition: .5s
    }

    .header_toggle {
    color: var(--first-color);
    font-size: 1.5rem;
    cursor: pointer
    }

    .header_img {
    width: 35px;
    height: 35px;
    display: flex;
    justify-content: center;
    border-radius: 50%;
    overflow: hidden
    }

    .header_img img {
    width: 40px
    }

    .l-navbar {
    position: fixed;
    top: 0;
    left: -30%;
    width: var(--nav-width);
    height: 100vh;
    background-color: var(--first-color);
    padding: .5rem 1rem 0 0;
    transition: .5s;
    z-index: var(--z-fixed)
    }


    h2 {
        margin-left: 6em;
    }

    img {
        margin-left: 5em;
    }

    .nav {
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow: hidden
    }

    .nav_logo,
    .nav_link {
    display: grid;
    grid-template-columns: max-content max-content;
    align-items: center;
    column-gap: 1rem;
    padding: .5rem 0 .5rem 1.5rem
    }

    .nav_logo {
    margin-bottom: 2rem
    }

    .nav_logo-icon {
    font-size: 1.25rem;
    color: var(--white-color)
    }

    .nav_logo-name {
    color: var(--white-color);
    font-weight: 700
    }

    .nav_link {
    position: relative;
    color: var(--first-color-light);
    margin-bottom: 1.5rem;
    transition: .3s
    }

    .nav_link:hover {
    color: var(--white-color)
    }

    .nav_icon {
    font-size: 1.25rem
    }

    .show {
    left: 0
    }

    .body-pd {
    padding-left: calc(var(--nav-width) + 1rem)
    }

    .active {
    color: var(--white-color)
    }

    .active::before {
    content: '';
    position: absolute;
    left: 0;
    width: 2px;
    height: 32px;
    background-color: var(--white-color)
    }

    .height-100 {
    height: 100vh
    }

    @media screen and (min-width: 768px) {
    body {
        margin: calc(var(--header-height) + 1rem) 0 0 0;
        padding-left: calc(var(--nav-width) + 2rem)
    }

    .header {
        height: calc(var(--header-height) + 1rem);
        padding: 0 2rem 0 calc(var(--nav-width) + 2rem)
    }

    .header_img {
        width: 40px;
        height: 40px
    }

    .header_img img {
        width: 45px
    }

    .l-navbar {
        left: 0;
        padding: 1rem 1rem 0 0
    }

    .show {
        width: calc(var(--nav-width) + 156px)
    }

    .body-pd {
        padding-left: calc(var(--nav-width) + 188px)
    }
    }
    </style>
    <script>
    document.addEventListener("DOMContentLoaded", function(event) {

    const showNavbar = (toggleId, navId, bodyId, headerId) =>{
    const toggle = document.getElementById(toggleId),
    nav = document.getElementById(navId),
    bodypd = document.getElementById(bodyId),
    headerpd = document.getElementById(headerId)

    // Validate that all variables exist
    if(toggle && nav && bodypd && headerpd){
    toggle.addEventListener('click', ()=>{
    // show navbar
    nav.classList.toggle('show')
    // add padding to body
    bodypd.classList.toggle('body-pd')
    // add padding to header
    headerpd.classList.toggle('body-pd')
    })
    }
    }

    showNavbar('header-toggle','nav-bar','body-pd','header')

    /*===== LINK ACTIVE =====*/
    const linkColor = document.querySelectorAll('.nav_link')

    function colorLink(){
    if(linkColor){
    linkColor.forEach(l=> l.classList.remove('active'))
    this.classList.add('active')
    }
    }
    linkColor.forEach(l=> l.addEventListener('click', colorLink))

    // Your code to run since DOM is loaded and ready
    });
    </script>'''    
            f.write(text)
                    
            f.close()