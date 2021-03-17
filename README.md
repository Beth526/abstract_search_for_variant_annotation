# abstract_search_for_variant_annotation
neural networks to score PubMed search results for similarity to evidence abstracts from ClinVar or CIViC 

The goal of this project is to make an app that will search PubMed for abstracts of interest, and then return the results scored for similarity to either abstracts from ClinVar or CIViC. This would be useful for variant annotation, because users could search a gene, disease name or drug, and easily find abstracts relevant to annotation efforts. 

ClinVar focuses on classifying germline mutations as benign or pathogenic for disease, and CIViC focuses on gathering prognostic, functional, and effect on therapy response for somatic mutations in cancer.

I didn't include journal information for predictions, there are a lot of journals but some are clearly more represented
![Top ClinVar Journals](https://github.com/Beth526/abstract_search_for_variant_annotation/blob/main/images/JournalPieChart.jpeg)

The streamlit app is working with the given files and requirements.txt, and I want to make a docker container for it and try to deploy it. The app shows the top relevant articles from the search and allows users to download the full table, including abstrast, title, journal, pmid, and score columns.

### Test set results ClinVar model
1 = abstract from ClinVar
0 = random abstract from 2018

![Test set results ClinVar model](https://github.com/Beth526/abstract_search_for_variant_annotation/blob/main/images/ClinVar%20model%20stats.png)

### Test set results CIViC model
1 = abstract from CIViC
0 = abstract with MESH terms Cancer, Protein or Gene from 2018

![Test set results CIViC model](https://github.com/Beth526/abstract_search_for_variant_annotation/blob/main/images/CIViC%20model%20stats.png)

The model also predicts quickly because it's pretty small. I first tried using DistilBERT embeddings but that was slow and didn't contain a lot of tokens in scientific abstracts. Instead I used a word2vec model I made earlier with 300k PubMed abstracts for Metis project 3. 
