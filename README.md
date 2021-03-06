# abstract_search_for_variant_annotation
### neural networks to score PubMed search results for similarity to evidence abstracts from ClinVar or CIViC 

The goal of this project is to create an app that will search PubMed for abstracts of interest, and then return the results scored for similarity to either abstracts from ClinVar or CIViC databases. This would be useful for variant annotation, because users could search a gene, disease name or drug, and easily find abstracts relevant to annotation. I removed gene and gene product names with a NER before training the model to try to reduce bias toward certain genes. 

ClinVar focuses on classifying germline mutations as benign or pathogenic for disease, and CIViC focuses on gathering prognostic, functional, and effect on therapy response for somatic mutations in cancer.

I put this app into a docker container: https://hub.docker.com/r/beth526/abstract_app
And you can run it with 'docker run -p 8501:8501 beth526/abstract_app' 

Searches can be things like 'ion channel AND epilepsy', 'polycystic kidney disease', 'FGFR AND bladder cancer', 'olaparib OR rucaparib AND BRCA1'

Journal information is currently not included for predictions, but some are clearly more represented:
![Top ClinVar Journals](https://github.com/Beth526/abstract_search_for_variant_annotation/blob/main/images/JournalPieChart.jpeg)

### Test set results ClinVar model
- 1 = abstract from ClinVar
- 0 = random abstract from 2018

![Test set results ClinVar model](https://github.com/Beth526/abstract_search_for_variant_annotation/blob/main/images/ClinVar%20model%20stats.png)

### Test set results CIViC model
- 1 = abstract from CIViC
- 0 = abstract with MESH terms Cancer, Protein or Gene from 2018

![Test set results CIViC model](https://github.com/Beth526/abstract_search_for_variant_annotation/blob/main/images/CIViC%20model%20stats.png)

The model also predicts quickly because it's pretty small. I first tried using DistilBERT embeddings but that was slow and didn't contain a lot of tokens in scientific abstracts. Instead I used a word2vec model I made earlier with 300k PubMed abstracts for Metis project 3. 

The app returns information on the top scoring articles under the ClinVar or CIViC model and allows users to download the full table, including abstrast, title, journal, pmid, and score columns.

Screenshot part 1:
![App part 1](https://github.com/Beth526/abstract_search_for_variant_annotation/blob/main/images/App%20top%20of%20page.png)

Screenshot part 2:
![App part 2](https://github.com/Beth526/abstract_search_for_variant_annotation/blob/main/images/App%20bottom%20of%20page.png)
