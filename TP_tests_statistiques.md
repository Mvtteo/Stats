# TP — Introduction aux Tests Statistiques

### Comparaison de deux architectures de deep learning en imagerie medicale

---

> **Comment utiliser ce TP**
>
> Ce TP se fait en binome. Chaque question appelle une reponse ecrite dans l'espace prevu, puis une verification en Python. L'objectif n'est pas d'arriver vite au code : c'est de comprendre ce que vous calculez avant de l'executer. Lisez chaque section en entier avant de repondre aux questions.

---

## Contexte du projet

Vous travaillez dans une startup de sante qui developpe un outil d'aide au diagnostic pour les radiologues. Votre equipe a entraine deux architectures de reseaux de neurones sur un meme jeu de donnees de **radiographies thoraciques** (tache de classification binaire : pneumonie / sain).

Les deux modeles ont ete entraines chacun **30 fois** avec des initialisations aleatoires differentes (seeds differentes). Cela permet d'evaluer non seulement la performance moyenne de chaque architecture, mais aussi sa **stabilite** : un modele tres performant en moyenne mais tres variable d'un run a l'autre est risque en production medicale, car on ne sait pas quel "version" du modele on obtiendra lors du deploiement reel.

Les deux architectures testees sont :

- **ResNet-50** : architecture profonde a connexions residuelles, souvent tres performante sur les taches de vision medicale, mais plus lourde en memoire GPU.
- **EfficientNet-B0** : architecture optimisee pour etre legere et rapide, conçue pour fonctionner sur des machines moins puissantes.

Pour chaque run d'entrainement, quatre metriques ont ete enregistrees :

| Colonne           | Description                                                                |
| ----------------- | -------------------------------------------------------------------------- |
| `accuracy`      | Proportion de radiographies correctement classifiees                       |
| `f1_score`      | Equilibre precision/rappel (crucial quand les classes sont desequilibrees) |
| `inference_ms`  | Temps moyen pour classer une image (en millisecondes)                      |
| `gpu_memory_mb` | Memoire GPU consommee pendant l'inference (en Mo)                          |

Le fichier de donnees s'appelle **`resultats_modeles.csv`**. Vous le chargez ainsi :

```python
import pandas as pd
df = pd.read_csv("resultats_modeles.csv")
```

---

## Partie 1 — Explorer les donnees avant tout calcul

Avant de lancer le moindre test statistique, un bon data scientist explore toujours ses donnees a la main. Cette etape s'appelle l'**analyse exploratoire** (EDA pour Exploratory Data Analysis). Elle permet de detecter des anomalies, de comprendre la forme des distributions, et de ne pas etre surpris par les resultats des tests. Un test statistique produit toujours un resultat — meme sur des donnees absurdes. C'est a vous de juger si ce resultat a du sens.

---

### 1.1 — Premier regard sur les donnees

Chargez le fichier CSV et affichez les premieres lignes ainsi qu'un resume statistique par modele.

**Fonctions utiles :** `df.head()`, `df.describe()`, `df.groupby("modele").describe()`

**Question 1.1.a — Combien de lignes et de colonnes contient le fichier ? Qu'est-ce que chaque ligne represente exactement ?**

```
Votre reponse :



```

**Question 1.1.b — Sans encore calculer quoi que ce soit, quel modele vous semble meilleur en termes d'accuracy ? Quelle est la difference de moyenne observee entre les deux modeles ?**

```
Votre reponse :



```

**Question 1.1.c — Cette difference observee vous semble-t-elle grande ou petite ? Pourquoi ne peut-on pas conclure directement a partir de cette difference ?**

> **Piste de reflexion :** pensez a la variabilite des runs. Si les deux modeles varient beaucoup d'un run a l'autre, la difference de moyenne peut etre due au hasard de l'initialisation aleatoire plutot qu'a une vraie superiorite d'architecture. En d'autres termes : si vous reprenez ces 30 runs avec 30 autres seeds, obtiendriez-vous exactement la meme difference ?

```
Votre reponse :



```

---

### 1.2 — Visualiser les distributions

Tracez des histogrammes et des boxplots pour les deux modeles sur la colonne `accuracy`. Superposez les deux modeles sur le meme graphique pour faciliter la comparaison.

**Fonctions utiles :** `matplotlib.pyplot` — `plt.hist()`, `plt.boxplot()` — ou `seaborn` — `sns.histplot()`, `sns.boxplot()`

**Question 1.2.a — En regardant vos histogrammes, les distributions vous semblent-elles symetriques ? Est-ce qu'elles ressemblent a des cloches ? Y a-t-il des valeurs qui semblent isolees du reste (valeurs aberrantes) ?**

```
Votre reponse :



```

**Question 1.2.b — Sur les boxplots, comparez les medianes et les etendues interquartiles (la hauteur de la boite) des deux modeles. Qu'est-ce que l'etendue interquartile vous dit sur la stabilite d'un modele ?**

> **Rappel :** l'etendue interquartile (IQR) est la distance entre le 1er quartile (Q1) et le 3e quartile (Q3). Elle capture la dispersion des 50% de valeurs centrales, sans etre influencee par les valeurs extremes. Un IQR petit signifie que les 30 runs donnent des resultats tres proches les uns des autres.

```

Votre reponse :



```

**Question 1.2.c — D'apres vos visualisations, quel modele vous semble le plus stable ? Est-ce le meme que celui qui a la meilleure accuracy moyenne ?**

```
Votre reponse :



```

---

### 1.3 — Calculer les statistiques descriptives

Calculez pour chaque modele et pour la colonne `accuracy` : la moyenne, l'ecart-type, le minimum, le maximum et la mediane.

**Fonctions utiles :** `.mean()`, `.std(ddof=1)`, `.min()`, `.max()`, `.median()`

> **Pourquoi `ddof=1` dans `.std()` ?** Par defaut, Python calcule l'ecart-type de la population entiere en divisant par n. Mais ici vous avez un echantillon de 30 runs parmi tous les runs possibles. La correction de Bessel (`ddof=1`) divise par (n-1) au lieu de n, ce qui produit un estimateur non biaise de l'ecart-type de la population. En pratique, la difference est faible pour n=30, mais c'est la bonne habitude a prendre systematiquement.

**Question 1.3.a — Remplissez ce tableau avec vos resultats :**

| Statistique         | ResNet-50 | EfficientNet-B0 |
| ------------------- | --------- | --------------- |
| Moyenne             |           |                 |
| Ecart-type (ddof=1) |           |                 |
| Minimum             |           |                 |
| Maximum             |           |                 |
| Mediane             |           |                 |

**Question 1.3.b — L'ecart-type du ResNet-50 est-il plus grand ou plus petit que celui de l'EfficientNet-B0 ? Qu'est-ce que cela signifie concretement pour un radiologue qui utiliserait ce modele en production ?**

```
Votre reponse :



```

---

## Partie 2 — Verifier les conditions du test

Avant d'appliquer un test statistique, il faut toujours verifier que les hypotheses d'utilisation de ce test sont remplies. C'est une etape critique que beaucoup negligent : un test statistique applique sur des donnees qui violent ses hypotheses peut produire des p-values completement fausses, et donc des conclusions erronees.

Le **t-test de Welch** que nous allons utiliser dans la Partie 3 repose sur deux conditions :

1. Les donnees de chaque groupe sont **approximativement normalement distribuees** (ou n est suffisamment grand pour que le Theoreme Central Limite s'applique — en pratique, n >= 30 suffit souvent).
2. Les observations sont **independantes** les unes des autres (un run n'influence pas les autres).

La condition 2 est satisfaite par construction : chaque run utilise une seed differente et un entrainement completement independant.

La condition 1 se verifie avec le **test de Shapiro-Wilk**.

---

### 2.1 — Comprendre le test de Shapiro-Wilk

Le test de Shapiro-Wilk est lui-meme un test statistique. Ses hypotheses sont :

- **H0 :** les donnees suivent une loi normale.
- **H1 :** les donnees ne suivent pas une loi normale.

Notez que dans ce contexte particulier, **rejeter H0 est une mauvaise nouvelle** : cela signifie que la distribution n'est pas normale et que le t-test serait inapproprie.

Si la p-value du test de Shapiro-Wilk est superieure a 0.05, on ne rejette pas H0 : les donnees sont compatibles avec une distribution normale. On peut alors proceder au t-test.

**Question 2.1 — Avant d'executer le test, pourquoi est-il logique de supposer que les accuracies d'un modele de deep learning suivent approximativement une loi normale sur 30 runs independants ?**

> **Piste :** l'accuracy est calculee comme la somme de resultats binaires (correct = 1, incorrect = 0) divisee par le nombre total d'images du jeu de test. Que dit le Theoreme Central Limite sur la distribution d'une somme d'un grand nombre de variables independantes ?

```
Votre reponse :



```

---

### 2.2 — Appliquer le test de Shapiro-Wilk

Separez les donnees des deux modeles dans deux Series pandas, puis appliquez le test de Shapiro-Wilk sur la colonne `accuracy` de chaque modele.

**Fonctions utiles :**

- Filtrer par modele : `df[df["modele"] == "ResNet-50"]["accuracy"]`
- Importer scipy : `from scipy import stats`
- Executer le test : `stat, p = stats.shapiro(data)`
- La fonction retourne deux valeurs : la statistique W (proche de 1 = tres normal) et la p-value.

**Question 2.2.a — Notez les resultats obtenus :**

| Modele          | Statistique W | P-value | Les donnees sont normales ? |
| --------------- | ------------- | ------- | --------------------------- |
| ResNet-50       |               |         |                             |
| EfficientNet-B0 |               |         |                             |

**Question 2.2.b — Peut-on proceder au t-test ? Justifiez votre reponse en vous appuyant sur les p-values obtenues et la regle de decision au seuil 0.05.**

```
Votre reponse :



```

**Question 2.2.c — Que feriez-vous si le test de Shapiro-Wilk avait rejete la normalite pour l'un des deux groupes ?**

> **Indication :** il existe des tests dits "non parametriques" qui ne supposent pas la normalite. Le plus courant pour comparer deux groupes independants est le **test de Mann-Whitney U** (`stats.mannwhitneyu` dans scipy). Il est moins puissant que le t-test quand les conditions du t-test sont remplies, mais reste valide meme sans normalite.

```
Votre reponse :



```

---

## Partie 3 — Formuler et conduire le test statistique

Vous avez explore les donnees et verifie les conditions. Il est temps de formaliser la question de recherche sous forme d'hypotheses statistiques, puis de conduire le test.

---

### 3.1 — Formuler les hypotheses

Rappel de la situation : vous voulez savoir si la difference d'accuracy observee entre ResNet-50 et EfficientNet-B0 est reelle, ou si elle pourrait etre due au simple hasard des initialisations aleatoires.

**Question 3.1 — Formulez les hypotheses H0 et H1 pour ce test. Precisez aussi le type de test (unilateral ou bilateral) et justifiez votre choix.**

> **Choix unilateral vs bilateral :** un test bilateral (H1 : mu_A =/= mu_B) teste si les deux groupes sont differents dans n'importe quel sens. Un test unilateral (H1 : mu_A > mu_B) teste si un groupe est specifiquement superieur. La regle : le type de test doit etre choisi AVANT de regarder les donnees. Si vous choisissez un test unilateral apres avoir constate que ResNet est meilleur, vous biaisiez votre analyse.

```
H0 :


H1 :


Type de test choisi :


Justification :


```

---

### 3.2 — Comprendre la statistique t avant de la calculer

Le t-test de Welch calcule une statistique t selon la formule suivante :

```
        moyenne_A - moyenne_B
t = -------------------------------------------
    racine( variance_A/n_A  +  variance_B/n_B )
```

Le numerateur est simplement la difference observee entre les deux groupes. Le denominateur est l'erreur standard de cette difference : il mesure l'incertitude sur notre estimation. Plus les echantillons sont grands et peu variables, plus ce denominateur est petit — et donc plus t est grand pour une meme difference observee.

**Question 3.2.a — En utilisant les statistiques descriptives de la Partie 1, calculez a la main une estimation de la statistique t. Montrez chaque etape du calcul.**

```
Numerateur (difference des moyennes) :


Denominateur (erreur standard) :


Valeur de t estimee :

```

**Question 3.2.b — Que signifie intuitivement un t grand en valeur absolue (par exemple t = 4) ? Et un t proche de 0 (par exemple t = 0.3) ?**

```
Votre reponse :



```

**Question 3.2.c — Dans un test bilateral, travaille-t-on avec t ou avec |t| (valeur absolue) ? Pourquoi ?**

```
Votre reponse :



```

---

### 3.3 — Executer le t-test de Welch

Appliquez le t-test de Welch sur la colonne `accuracy` des deux groupes.

**Fonction a utiliser :** `stats.ttest_ind(groupe_A, groupe_B, equal_var=False)`

Le parametre `equal_var=False` specifie qu'on utilise la version de Welch, qui ne suppose pas que les deux groupes ont la meme variance. C'est le choix le plus sur par defaut : si les variances sont effectivement egales, la version de Welch donne des resultats quasi-identiques au t-test classique. L'inverse n'est pas vrai.

La fonction retourne deux valeurs : `(statistique_t, p_value)`.

**Question 3.3.a — Notez les resultats obtenus :**

| Statistique t | P-value |
| ------------- | ------- |
|               |         |

**Question 3.3.b — La valeur de t que vous obtenez est-elle proche de votre estimation manuelle de la question 3.2.a ? Si elle differe, pourquoi ?**

```
Votre reponse :



```

**Question 3.3.c — Interpretez la p-value obtenue en une phrase concrete. Que signifie-t-elle dans le contexte de votre comparaison de modeles ?**

> **Rappel :** la p-value est la probabilite d'observer une difference aussi grande (ou plus grande) entre les deux groupes SI H0 etait vraie, c'est-a-dire si les deux architectures avaient en realite exactement les memes performances moyennes dans la population.

```
Votre reponse :



```

**Question 3.3.d — Au seuil alpha = 0.05, quelle est votre decision statistique ? Formulez ensuite une conclusion en langage naturel, comme si vous l'ecriviez dans un rapport destine a un radiologue non statisticien.**

```
Decision statistique (rejet ou non rejet de H0) :


Conclusion en langage naturel :




```

---

## Partie 4 — La taille d'effet : ce que la p-value ne dit pas

Vous venez de repondre a la question "Y a-t-il une difference statistiquement significative ?". Mais cette question n'est qu'a moitie utile. La vraie question pour un deploiement en production medicale est : **"Cette difference est-elle assez grande pour valoir le cout de choisir un modele plutot que l'autre ?"**

Un test statistique avec n=10 000 observations peut detecter une difference de 0.001% en accuracy comme "significative". Cette difference est reelle — mais est-elle utile ? C'est precisement ce que mesure la **taille d'effet**.

---

### 4.1 — Le Cohen's d

Le Cohen's d est la mesure de taille d'effet la plus utilisee pour comparer deux moyennes. Il exprime la difference entre les deux groupes en unites d'ecart-type, ce qui la rend interpretable independamment de l'echelle de mesure :

```
         moyenne_A - moyenne_B
d = ---------------------------------
         ecart-type commun
```

Ou l'ecart-type commun (pooled) est : `s_pool = racine( (s_A^2 + s_B^2) / 2 )`

La regle de Cohen pour interpreter |d| :

| Valeur de |d| | Interpretation |
|-------------|----------------|
| < 0.2 | Effet negligeable |
| 0.2 a 0.5 | Petit effet |
| 0.5 a 0.8 | Effet moyen |
| > 0.8 | Grand effet |

**Question 4.1.a — Calculez le Cohen's d a partir de vos statistiques descriptives. Montrez le calcul etape par etape.**

```
s_pool = racine( (s_A^2 + s_B^2) / 2 ) =


d = (moyenne_A - moyenne_B) / s_pool =


Interpretation selon la regle de Cohen :

```

**Question 4.1.b — Traduisez ce resultat concretement : si un hopital utilise ResNet-50 plutot qu'EfficientNet-B0 et traite 1000 radiographies, combien de diagnostics supplementaires seront correctement emis (en supposant des conditions identiques) ?**

> **Methode :** la difference de moyenne en accuracy multipliee par le nombre d'images donne le nombre de cas supplementaires correctement classes.

```
Calcul :


Reponse :

```

---

### 4.2 — Une p-value significative avec un effet negligeable

Voici un phenomene contre-intuitif que tout praticien ML doit avoir vu au moins une fois. Imaginez que la vraie difference entre ResNet-50 et EfficientNet-B0 ne soit pas de ~2% en accuracy, mais seulement de **0.1%** (un dixieme de pourcent). C'est une difference totalement negligeable en pratique medicale. Pourtant, avec suffisamment de donnees, le test statistique la detectera comme "significative".

Pour le demontrer, generez deux groupes avec une difference de moyenne de 0.001 et un ecart-type de 0.02, et faites varier la taille d'echantillon.

**Fonctions utiles :** `np.random.normal(loc, scale, size)`, boucle `for n in [10, 100, 1000, 10000, 100000]`

**Question 4.2.a — Pour chaque valeur de n ci-dessous, notez la p-value obtenue et indiquez si la difference est "significative" (p < 0.05) :**

| n par groupe | P-value obtenue | Significatif ? |
| ------------ | --------------- | -------------- |
| 10           |                 |                |
| 100          |                 |                |
| 1 000        |                 |                |
| 10 000       |                 |                |
| 100 000      |                 |                |

**Question 4.2.b — Calculez le Cohen's d pour cette difference de 0.001 avec un ecart-type de 0.02. Que vaut-il ? Est-ce que sa valeur change quand n augmente ?**

```
d = 0.001 / 0.02 =


Change-t-il avec n ? Pourquoi ?


```

**Question 4.2.c — Quelle lecon tirez-vous de cet exercice pour la pratique du deep learning, notamment quand on compare des modeles sur de tres grands jeux de donnees ?**

```
Votre reponse :




```

---

## Partie 5 — Aller plus loin : analyser toutes les metriques

Jusqu'ici vous avez compare les deux modeles sur l'accuracy. Mais votre fichier contient trois autres colonnes : `f1_score`, `inference_ms` et `gpu_memory_mb`. Un bon choix d'architecture en production prend en compte tous ces criteres simultanement.

---

### 5.1 — Repeter l'analyse sur le f1_score

L'accuracy mesure le taux global de bonnes classifications. Mais dans le contexte medical, les erreurs ne se valent pas : rater une pneumonie (faux negatif) est bien plus grave que classer a tort un poumon sain comme malade (faux positif). Le **F1-score** equilibre ces deux types d'erreurs et est souvent plus pertinent que l'accuracy en clinique.

**Question 5.1.a — Repetez les etapes des Parties 2 et 3 sur la colonne `f1_score`. Remplissez le tableau suivant :**

| Etape                | ResNet-50 | EfficientNet-B0 |
| -------------------- | --------- | --------------- |
| Moyenne f1_score     |           |                 |
| Ecart-type f1_score  |           |                 |
| Shapiro-Wilk p-value |           |                 |
| Normalite verifiee ? |           |                 |

| Resultat du test       | Valeur |
| ---------------------- | ------ |
| Statistique t          |        |
| P-value                |        |
| Decision au seuil 0.05 |        |
| Cohen's d              |        |

**Question 5.1.b — Les conclusions sur le f1_score sont-elles coherentes avec celles sur l'accuracy ? Si les resultats different, comment l'interpretez-vous ?**

```
Votre reponse :



```

---

### 5.2 — Le temps d'inference : une metrique critique en production

En milieu hospitalier, le temps de reponse d'un systeme d'aide au diagnostic peut avoir un impact direct sur le flux de travail. Une difference de quelques millisecondes par image devient significative si le systeme traite des milliers de radiographies par jour.

**Question 5.2.a — Sans executer de test, regardez les moyennes d'`inference_ms` pour les deux modeles. Laquelle est la plus rapide ? Quelle est la difference en millisecondes ?**

```
Votre reponse :



```

**Question 5.2.b — Conduisez le t-test sur `inference_ms`. Notez les resultats.**

```
Statistique t :
P-value :
Decision au seuil 0.05 :
Cohen's d :

```

**Question 5.2.c — Si un hopital traite 5000 radiographies par jour, quelle est la difference de temps total de traitement entre les deux modeles en utilisant les moyennes observees ? Convertissez en minutes. Cette difference vous semble-t-elle operationnellement significative ?**

```
Calcul :




Reponse :

```

---

### 5.3 — Synthese : quelle architecture recommanderiez-vous ?

Vous avez maintenant analyse quatre metriques. La decision finale n'est pas purement statistique : elle depend aussi des contraintes operationnelles et des priorites cliniques du contexte de deploiement.

**Question 5.3.a — Remplissez ce tableau de synthese :**

| Metrique          | Meilleur modele | Difference significative ? | Cohen's d | Interpretation |
| ----------------- | --------------- | -------------------------- | --------- | -------------- |
| Accuracy          |                 |                            |           |                |
| F1-score          |                 |                            |           |                |
| Temps d'inference |                 |                            |           |                |
| Memoire GPU       |                 |                            |           |                |

**Question 5.3.b — Formulez votre recommandation finale (5 a 8 lignes). Precisez dans quel contexte vous privilegieriez chaque modele. Par exemple : hopital universitaire avec serveur GPU haut de gamme vs. clinique rurale avec budget informatique limite.**

```
Votre recommandation :




```

---

## Aide-memoire — recapitulatif des fonctions utilisees

| Etape                         | Fonction Python                                                      | Ce qu'elle retourne       |
| ----------------------------- | -------------------------------------------------------------------- | ------------------------- |
| Charger les donnees           | `pd.read_csv("resultats_modeles.csv")`                             | DataFrame                 |
| Stats descriptives par groupe | `df.groupby("modele").describe()`                                  | Tableau de statistiques   |
| Test de normalite             | `stats.shapiro(data)` retourne `(W, p_value)`                    | W proche de 1 = normal    |
| T-test de Welch               | `stats.ttest_ind(A, B, equal_var=False)` retourne `(t, p_value)` | Test bilateral par defaut |
| Alternative non parametrique  | `stats.mannwhitneyu(A, B, alternative="two-sided")`                | Si normalite violee       |
| Histogramme                   | `plt.hist(data, bins=20, alpha=0.6)`                               | Graphique                 |
| Boxplot compare               | `df.boxplot(column="accuracy", by="modele")`                       | Graphique                 |

---

## Bareme indicatif

| Partie                                               | Points           |
| ---------------------------------------------------- | ---------------- |
| Partie 1 — Exploration et statistiques descriptives | 4 pts            |
| Partie 2 — Verification des conditions              | 3 pts            |
| Partie 3 — Formulation et conduite du test          | 5 pts            |
| Partie 4 — Taille d'effet et interpretation         | 4 pts            |
| Partie 5 — Analyse multi-metriques et synthese      | 4 pts            |
| **Total**                                      | **20 pts** |

Les points sont attribues en priorite aux **justifications ecrites**, pas aux resultats numeriques. Un resultat correct sans explication ne rapporte que la moitie des points. Une bonne explication accompagnant un resultat legerement faux rapporte les trois quarts des points.

---

*TP concu pour des etudiants ayant des bases en Python et en machine learning. Duree estimee : 2h30.*
