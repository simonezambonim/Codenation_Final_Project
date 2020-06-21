#Final Project

## Objective
The purpose of this product is to provide an automated service that recommends leads to a user given their current customer list (Portfolio).

## Contextualization
Some companies would like to know who are the other companies in a given market (population) that are most likely to become their next customers. That is, your solution must find in the market who are the most adherent leads given the characteristics of the customers present in the user's portfolio.

In addition, your solution must be user agnostic. Any user with a list of customers who want to explore this market can extract value from the service.

For the challenge, the following datasets should be considered:

Market: Base with information about the companies in the market to be considered. 
Portfolio 1: Company customer ids 1 
Portfolio 2: Company customer ids 2 
Portfolio 3: Company customer ids 3

Note: all companies (ids) in the portfolios are contained in the Market (population base).

## Content

Here are some information to help you get started. 

* folder ```data``` - contains the 3 portfolios and the market dataset
* folder ```app``` - contains the recommender system (:warning: remember to change the variable ```file_path``` in main.py and recommender.py) 
* folder ```output``` - contains the entire leads of recommended leads for each portfolio

### Prerequisites

The packages needed to run the app are in the requirements.txt file.

### Installing

Clone the repository and install all the packages necessary:

```
cd path
vintualenv venv
cd path\venv\Scripts\activate

pip install -r requirements.txt 
```

Go to the path folder ```cd .../app```Use the following command to run the application:

```
streamlit run main.py
```

## Built With
* [Gower](https://github.com/wwwjk366/gower)
* [HEOM](https://github.com/KacperKubara/distython)
* [Streamlit](https://docs.streamlit.io/api.html) - The web framework 
* [Plotly express](https://plotly.com/python/plotly-express/) - Interactive plots


## Author

**Simone Rosana Zambonim**  - [Linkedin](https://www.linkedin.com/in/simonezambonim/) [Github](https://github.com/simonezambonim/)


## Acknowledgments

 ### Gower coefficient
 [Michal Yan - `!pip install gower` ](https://www.thinkdatascience.com/post/2019-12-16-introducing-python-package-gower/)
 [Marcelo Beckmann](https://sourceforge.net/projects/gower-distance-4python/files/)
 ### HEOM - Heterogeneous Euclidean-Overlap Metric
 [Kacper Kubara - !pip install distython](https://towardsdatascience.com/distython-5de10f342c93)
 [Wilson and Martinez(1997)](https://arxiv.org/pdf/cs/9701101.pdf)
 
 
 



