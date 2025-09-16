# Atividades de Sala
> Orientações para execução das atividades de sala.

Esse documento exibe as descrições das questões a serem resolvidas em sala


### Questão 1

[1_countVectorizer.py](1_countVectorizer.py)

#### Utilização da CountVectorizer 

  
#### Instruções:

1) importe ``CountVectorizer`` e ``train_test_split``.
   
2) Instancie CountVectorizer, use stopwords english.

3) Transforme os dados de treino
   
4) utilizando a função ``get_feature_names_out()`` do objeto ``count_vectorizer``  mostre os atributos selecionados.


### Questão 2

[2_tf-idf-sklearn.py](2_tf-idf-sklearn.py)

#### Extração de atributos com sklearn tfidf

#### Instruções 
  
1) importe ``TfidfVectorizer`` da scikit-learn.
   
2) Instancie TF-IDF, use stopwords english.

3) Transforme os dados de treino
   
4) utilizando a função ``get_feature_names_out()`` do objeto ``tfidf_vectorizer``  mostre os atributos selecionados.


### Questão 3

[3_fake_news_classifier_count_vectorizer.py](3_fake_news_classifier_count_vectorizer.py)

#### Classificação com o atributo countVectorizer

#### Instruções
  
1) Instancie o naive bayes.
2) Execute a função fit no conjunto de treino.
3) Execute a predição no conjunto de teste.
4) Compute a acurácia
5) plote a atriz de confusão

### Questão 4

[4_fake_news_classifier_tfidf.py](4_fake_news_classifier_tfidf.py)

#### Instruções

1) Instancie o naive bayes.
2) Execute a função fit no conjunto de treino.
3) Execute a predição no conjunto de teste.
4) Compute a acurácia
5) plote a atriz de confusão