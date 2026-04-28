# Cours de Statistiques — Tests Statistiques
### Destiné aux étudiants en Machine Learning & Deep Learning

---

## Table des matières

1. [Les distributions statistiques fondamentales](#1-les-distributions-statistiques-fondamentales)
2. [Qu'est-ce qu'un test statistique ?](#2-quest-ce-quun-test-statistique-)
3. [La p-value — définition et interprétation](#3-la-p-value--définition-et-interprétation)
4. [La démarche complète en 5 étapes](#4-la-démarche-complète-en-5-étapes)
5. [Les tests statistiques courants](#5-les-tests-statistiques-courants)
6. [Erreurs classiques à éviter](#6-erreurs-classiques-à-éviter)
7. [Implémentation Python](#7-implémentation-python)
8. [Aide-mémoire — quel test choisir ?](#8-aide-mémoire--quel-test-choisir-)

---

## 1. Les distributions statistiques fondamentales

Une **distribution** décrit comment les valeurs d'une variable aléatoire se répartissent. Avant de choisir un test, il faut comprendre quelle distribution sous-tend les données ou la statistique de test.

---

### 1.1 La loi Normale — N(μ, σ²)

La loi la plus fondamentale en statistique. Elle est caractérisée par sa forme en cloche symétrique.

**Paramètres :**
- `μ` (mu) : la moyenne — centre de la distribution
- `σ` (sigma) : l'écart-type — largeur de la cloche

**Formule de la densité :**

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Pourquoi est-elle omniprésente ?**
Le **Théorème Central Limite (TCL)** garantit que la somme (ou moyenne) d'un grand nombre de variables aléatoires indépendantes converge vers une loi normale, quelle que soit la distribution d'origine.

> **En ML/DL :** résidus de régression, initialisation des poids (Glorot/He), bruit gaussien dans les autoencodeurs variationnels (VAE), hypothèse de bruit dans les modèles linéaires.

**Règle empirique (règle des 68-95-99.7) :**
- ±1σ contient environ **68%** des données
- ±2σ contient environ **95%** des données
- ±3σ contient environ **99.7%** des données

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

mu, sigma = 0, 1
x = np.linspace(-4, 4, 300)
y = stats.norm.pdf(x, mu, sigma)

plt.plot(x, y)
plt.fill_between(x, y, where=(np.abs(x) <= 1), alpha=0.3, label='±1σ (68%)')
plt.fill_between(x, y, where=(np.abs(x) <= 2), alpha=0.2, label='±2σ (95%)')
plt.title('Loi Normale N(0, 1)')
plt.legend()
plt.show()
```

---

### 1.2 La loi de Student — t(ν)

La loi de Student est une loi normale "épaissie aux queues", conçue pour les petits échantillons dont on ne connaît pas σ.

**Paramètre :**
- `ν` (nu) : degrés de liberté. Quand ν → ∞, la loi t tend vers la loi normale.

**Intuition :** avec peu de données, on est moins certain de notre estimation de σ. La loi t est plus prudente : elle attribue plus de probabilité aux valeurs extrêmes.

**Règle pratique :** utiliser la loi t quand `n < 30` et σ inconnu. Au-delà, la normale est une bonne approximation.

> **En ML/DL :** t-test pour comparer les performances de deux modèles sur un petit nombre d'expériences, intervalles de confiance sur des métriques.

---

### 1.3 La loi Khi-deux — χ²(ν)

La somme de `ν` variables normales centrées réduites au carré suit une loi χ².

$$\chi^2(\nu) = Z_1^2 + Z_2^2 + \cdots + Z_\nu^2 \quad \text{où } Z_i \sim N(0,1)$$

**Propriétés :** toujours positive, asymétrique à droite. Quand ν augmente, elle devient de plus en plus symétrique et tend vers la normale.

> **En ML/DL :** test d'indépendance entre variables catégorielles (ex : type d'erreur selon le modèle), test du rapport de vraisemblance, régularisation L2 (connexion avec la distribution a priori gaussienne).

---

### 1.4 La loi de Fisher — F(d₁, d₂)

Rapport de deux variables χ² indépendantes divisées par leurs degrés de liberté.

$$F = \frac{\chi^2_{d_1}/d_1}{\chi^2_{d_2}/d_2}$$

**Paramètres :** `d₁` (numérateur) et `d₂` (dénominateur).

> **En ML/DL :** ANOVA pour comparer 3+ modèles, test global de significativité en régression (F-statistic), analyse de variance dans les expériences d'hyperparamétrage.

---

### 1.5 La loi Binomiale — B(n, p)

Modélise le nombre de succès en `n` essais indépendants, chacun avec probabilité de succès `p`.

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

> **En ML/DL :** taux d'erreur de classification (Bernoulli répété), tests A/B, bootstrap, dropout (masques aléatoires).

---

## 2. Qu'est-ce qu'un test statistique ?

Un test statistique est une **procédure formelle** permettant de décider, à partir d'un échantillon de données, si une hypothèse sur la population est plausible ou non.

### Vocabulaire essentiel

| Terme | Définition |
|-------|-----------|
| **H₀** (hypothèse nulle) | Ce qu'on cherche à réfuter — état de "rien de spécial" |
| **H₁** (hypothèse alternative) | Ce qu'on cherche à montrer |
| **Statistique de test** | Un nombre résumant l'écart entre les données et H₀ |
| **P-value** | Probabilité d'observer des données aussi extrêmes si H₀ était vraie |
| **Seuil α** | Risque qu'on accepte de se tromper (souvent 5%) |
| **Région de rejet** | Valeurs de la statistique incompatibles avec H₀ |

### Intuition fondamentale

> **On ne prouve jamais H₀. On cherche à montrer que les données lui sont incompatibles.**

Le raisonnement est indirect, comme un raisonnement par l'absurde :

1. On suppose H₀ vraie.
2. On calcule la probabilité d'obtenir les données observées (ou pire) sous cette hypothèse.
3. Si cette probabilité est très faible → les données sont incompatibles avec H₀ → on rejette H₀.

---

## 3. La p-value — définition et interprétation

### Définition exacte

> La **p-value** est la probabilité d'observer une statistique de test **aussi extrême ou plus extrême** que celle calculée, en supposant que H₀ est vraie.

$$p\text{-value} = P(|T| \geq |t_{obs}| \mid H_0 \text{ vraie})$$

### Ce que la p-value N'EST PAS

❌ La probabilité que H₀ soit vraie.  
❌ La probabilité que le résultat soit dû au hasard.  
❌ Une mesure de l'importance ou de la taille de l'effet.  
❌ Une preuve définitive.

### Ce que la p-value EST

✅ Une mesure de **compatibilité des données avec H₀**.  
✅ Plus elle est petite, plus les données sont suspectes sous H₀.  
✅ Un outil de décision, pas une vérité absolue.

### Interprétation pratique

| P-value | Interprétation |
|---------|----------------|
| p ≥ 0.10 | Données tout à fait compatibles avec H₀ |
| 0.05 ≤ p < 0.10 | Légère tendance, mais non significatif au seuil 5% |
| 0.01 ≤ p < 0.05 | Significatif au seuil 5% |
| 0.001 ≤ p < 0.01 | Très significatif |
| p < 0.001 | Extrêmement significatif |

### Tests unilatéraux vs bilatéraux

- **Test bilatéral** (H₁ : μ ≠ μ₀) : on s'intéresse aux deux queues — on ne sait pas dans quel sens l'effet va.
- **Test unilatéral droit** (H₁ : μ > μ₀) : on suppose que l'effet ne peut être que positif.
- **Test unilatéral gauche** (H₁ : μ < μ₀) : on suppose que l'effet ne peut être que négatif.

> **Règle :** choisir le type de test *avant* de regarder les données. Choisir un test unilatéral après avoir vu les données gonfle artificiellement la significativité.

---

## 4. La démarche complète en 5 étapes

### Étape 1 — Formuler les hypothèses

Définir H₀ et H₁ **avant** de regarder les données. H₀ doit être précise et testable.

**Exemples :**

| Situation | H₀ | H₁ |
|-----------|----|----|
| Le modèle A est-il meilleur que B ? | μ_A = μ_B | μ_A ≠ μ_B |
| L'accuracy dépasse-t-elle 90% ? | μ = 0.90 | μ > 0.90 |
| Les erreurs sont-elles indépendantes du type ? | indépendance | association |

---

### Étape 2 — Choisir le test approprié

Questions à se poser dans l'ordre :

1. **Quel type de données ?** Continues, catégorielles, ordinales ?
2. **Combien de groupes ?** 1, 2, ou plus ?
3. **Données indépendantes ou appariées ?**
4. **Les conditions de validité sont-elles remplies ?** (normalité, homogénéité des variances…)

> Voir le tableau récapitulatif en section 8.

---

### Étape 3 — Vérifier les conditions de validité

Chaque test repose sur des hypothèses. Les violer peut invalider les conclusions.

**Conditions fréquentes :**
- **Normalité** : vérifier avec Shapiro-Wilk (n < 50) ou D'Agostino-Pearson (n ≥ 50).
- **Homogénéité des variances** : vérifier avec le test de Levene.
- **Indépendance des observations** : vérifier conceptuellement (design de l'étude).
- **Effectifs suffisants** : pour le chi-deux, chaque cellule doit avoir un effectif théorique ≥ 5.

Si les conditions ne sont pas remplies → utiliser des alternatives non paramétriques (Mann-Whitney, Kruskal-Wallis, etc.).

---

### Étape 4 — Calculer la statistique de test et la p-value

La statistique de test mesure combien l'échantillon s'éloigne de ce qu'on attendrait sous H₀, en unités d'erreur standard.

**Exemple pour le t-test à 2 échantillons :**

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

Plus `|t|` est grand, plus les données s'éloignent de H₀.

---

### Étape 5 — Conclure

| Résultat | Conclusion |
|----------|-----------|
| p < α | On **rejette H₀** — les données sont incompatibles avec H₀ au seuil α |
| p ≥ α | On **ne peut pas rejeter H₀** — manque de preuves (≠ H₀ est vraie !) |

**Toujours rapporter :**
- La statistique de test et les degrés de liberté
- La p-value
- La **taille d'effet** (voir section 6)
- L'intervalle de confiance si possible

---

## 5. Les tests statistiques courants

### 5.1 T-test à 1 échantillon

**Quand l'utiliser :** comparer la moyenne d'un groupe à une valeur de référence connue.

- **H₀ :** μ = μ₀
- **Conditions :** normalité (ou n > 30 par le TCL)
- **Statistique :** t de Student avec n-1 degrés de liberté

**Exemple :** Le temps de réponse moyen de notre API est-il différent de 100 ms ?

```python
from scipy import stats
import numpy as np

data = np.array([102, 98, 110, 95, 107, 103, 99, 108, 101, 97])
mu0 = 100

t_stat, p_value = stats.ttest_1samp(data, popmean=mu0)
print(f"t = {t_stat:.4f}, p = {p_value:.4f}")
```

---

### 5.2 T-test à 2 échantillons indépendants (Welch)

**Quand l'utiliser :** comparer les performances de deux modèles sur des expériences indépendantes.

- **H₀ :** μ₁ = μ₂
- **Conditions :** normalité approximative (ou n > 30)
- **Version de Welch :** ne suppose pas l'égalité des variances — à préférer par défaut

```python
from scipy import stats
import numpy as np

modele_A = [0.92, 0.88, 0.91, 0.93, 0.90, 0.89, 0.94, 0.87, 0.92, 0.91]
modele_B = [0.85, 0.83, 0.87, 0.86, 0.84, 0.88, 0.82, 0.85, 0.86, 0.83]

# Vérification de la normalité (Shapiro-Wilk)
_, p_normA = stats.shapiro(modele_A)
_, p_normB = stats.shapiro(modele_B)
print(f"Normalité A: p={p_normA:.3f} | B: p={p_normB:.3f}")

# Vérification de l'égalité des variances (Levene)
_, p_var = stats.levene(modele_A, modele_B)
equal_var = p_var > 0.05  # False → Welch

# T-test de Welch
t_stat, p_value = stats.ttest_ind(modele_A, modele_B, equal_var=equal_var)

# Taille d'effet : Cohen's d
d = (np.mean(modele_A) - np.mean(modele_B)) / np.sqrt(
    (np.std(modele_A, ddof=1)**2 + np.std(modele_B, ddof=1)**2) / 2
)

print(f"t = {t_stat:.3f}, p = {p_value:.4f}, Cohen's d = {d:.3f}")
```

---

### 5.3 T-test apparié (avant/après)

**Quand l'utiliser :** les deux mesures proviennent du même sujet ou des mêmes données (même fold de validation, même batch de test).

- **H₀ :** μ_différence = 0
- **Avantage :** élimine la variabilité inter-individuelle → plus puissant que le t-test indépendant

```python
from scipy import stats

# Accuracy d'un même modèle avant/après fine-tuning sur les mêmes données
avant      = [0.81, 0.79, 0.83, 0.80, 0.78, 0.82, 0.80, 0.81]
apres      = [0.85, 0.84, 0.86, 0.83, 0.82, 0.87, 0.84, 0.85]

t_stat, p_value = stats.ttest_rel(avant, apres)
print(f"t = {t_stat:.3f}, p = {p_value:.4f}")
```

---

### 5.4 ANOVA à 1 facteur

**Quand l'utiliser :** comparer 3 modèles ou plus simultanément.

- **H₀ :** μ₁ = μ₂ = ... = μₖ (toutes les moyennes sont égales)
- **H₁ :** au moins un groupe diffère
- **Attention :** un résultat significatif dit seulement qu'il y a *une* différence quelque part. Il faut un test post-hoc pour identifier *laquelle*.

```python
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np

np.random.seed(42)
modele_A = np.random.normal(0.90, 0.02, 20)
modele_B = np.random.normal(0.87, 0.02, 20)
modele_C = np.random.normal(0.85, 0.02, 20)

# ANOVA
f_stat, p_anova = stats.f_oneway(modele_A, modele_B, modele_C)
print(f"ANOVA : F = {f_stat:.2f}, p = {p_anova:.4f}")

# Post-hoc Tukey (uniquement si p < 0.05)
if p_anova < 0.05:
    data   = np.concatenate([modele_A, modele_B, modele_C])
    groups = ['A'] * 20 + ['B'] * 20 + ['C'] * 20
    result = pairwise_tukeyhsd(data, groups, alpha=0.05)
    print(result)
```

---

### 5.5 Test du Chi-deux d'indépendance

**Quand l'utiliser :** tester si deux variables catégorielles sont liées.

- **H₀ :** les variables sont indépendantes
- **Conditions :** effectif théorique ≥ 5 dans chaque cellule

```python
from scipy import stats
import numpy as np

# Tableau de contingence : modèle × type d'erreur
# Lignes = modèles A, B, C | Colonnes = faux positif, faux négatif, correct
contingency = np.array([
    [45, 12,  8],   # Modèle A
    [30, 25, 15],   # Modèle B
    [50,  8, 12],   # Modèle C
])

chi2, p, ddl, expected = stats.chi2_contingency(contingency)
print(f"χ² = {chi2:.2f}, ddl = {ddl}, p = {p:.4f}")
print("Effectifs théoriques sous H₀ :")
print(expected.round(1))
```

---

### 5.6 Test de normalité — Shapiro-Wilk

**Quand l'utiliser :** vérifier si des données suivent une loi normale avant un t-test sur petit échantillon.

- **H₀ :** les données sont normalement distribuées
- **Recommandé pour n < 50.** Pour n ≥ 50, préférer le test de D'Agostino-Pearson.

```python
from scipy import stats
import numpy as np

def check_normality(data, alpha=0.05):
    """Choisit automatiquement le bon test selon n."""
    n = len(data)
    if n < 50:
        stat, p = stats.shapiro(data)
        test_name = "Shapiro-Wilk"
    else:
        stat, p = stats.normaltest(data)  # D'Agostino-Pearson
        test_name = "D'Agostino-Pearson"

    return {
        "test"      : test_name,
        "n"         : n,
        "statistic" : round(stat, 4),
        "p_value"   : round(p, 4),
        "normal"    : p > alpha
    }

residuals = np.random.normal(0, 1, 35)
print(check_normality(residuals))
```

---

## 6. Erreurs classiques à éviter

### 6.1 Les deux types d'erreurs

|  | H₀ vraie | H₀ fausse |
|--|----------|-----------|
| **On rejette H₀** | ❌ Erreur de type I (α) — faux positif | ✅ Bonne décision |
| **On ne rejette pas H₀** | ✅ Bonne décision | ❌ Erreur de type II (β) — faux négatif |

- **α** (risque de type I) : contrôlé par le seuil qu'on choisit (souvent 5%).
- **β** (risque de type II) : dépend de la taille d'effet et de n. La puissance du test = 1 − β.
- Il y a un compromis : réduire α augmente β. Pour les deux simultanément, il faut augmenter n.

---

### 6.2 Significativité statistique ≠ significativité pratique

> Avec un grand n, même un effet minuscule devient statistiquement significatif. Toujours rapporter la taille d'effet.

**Mesures de taille d'effet courantes :**

| Contexte | Mesure | Interprétation |
|----------|--------|----------------|
| Comparaison de moyennes | Cohen's d | < 0.2 négligeable, 0.2–0.5 petit, 0.5–0.8 moyen, > 0.8 grand |
| ANOVA | η² (eta carré) | proportion de variance expliquée |
| Corrélation | r de Pearson | −1 à 1 |
| Régression | R² | proportion de variance expliquée |
| Chi-deux | V de Cramér | 0 à 1 |

```python
import numpy as np

def cohen_d(group1, group2):
    """Calcule le Cohen's d pour deux groupes."""
    n1, n2 = len(group1), len(group2)
    s_pooled = np.sqrt(
        ((n1 - 1) * np.std(group1, ddof=1)**2 + (n2 - 1) * np.std(group2, ddof=1)**2)
        / (n1 + n2 - 2)
    )
    return (np.mean(group1) - np.mean(group2)) / s_pooled
```

---

### 6.3 Le problème des comparaisons multiples

Si on effectue 20 tests avec α = 0.05, on s'attend à obtenir **1 fausse positive par hasard** (0.05 × 20 = 1).

**Solutions :**
- **Correction de Bonferroni** : diviser α par le nombre de tests. Simple mais conservatrice.
- **Correction de Benjamini-Hochberg (FDR)** : contrôle le taux de fausses découvertes. Plus puissante.
- **Tests post-hoc** (Tukey, Scheffé) : pour l'ANOVA, ils intègrent automatiquement la correction.

```python
from statsmodels.stats.multitest import multipletests

# p-values brutes de plusieurs tests
p_values = [0.01, 0.04, 0.03, 0.20, 0.08, 0.02]

# Correction de Benjamini-Hochberg
reject, p_corrected, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)

for p_raw, p_corr, rej in zip(p_values, p_corrected, reject):
    print(f"p brut={p_raw:.3f} | p corrigé={p_corr:.3f} | Rejet={rej}")
```

---

### 6.4 Définir H₀ après avoir vu les données (p-hacking)

Adapter ses hypothèses ou son test *après* avoir vu les données fausse complètement le seuil α. C'est une pratique à proscrire.

**Règle :** définir H₀, H₁, le test et le seuil α **avant** de collecter ou d'analyser les données (pré-enregistrement).

---

### 6.5 Confondre "pas de différence significative" et "pas de différence"

Un test non significatif ne prouve pas que H₀ est vraie. Peut-être que l'effet existe mais que l'échantillon est trop petit pour le détecter (manque de puissance).

> Toujours vérifier la puissance du test, surtout pour conclure à l'absence d'effet.

---

## 7. Implémentation Python

### 7.1 Pipeline complet de test statistique

```python
from scipy import stats
import numpy as np
import pandas as pd

def full_test_pipeline(group1, group2, alpha=0.05):
    """
    Pipeline complet de comparaison de deux groupes.
    Sélectionne automatiquement le bon test selon les conditions.
    """
    results = {}
    
    # --- Étape 1 : Normalité ---
    n1, n2 = len(group1), len(group2)
    test_name = "Shapiro-Wilk"
    _, p_norm1 = stats.shapiro(group1) if n1 < 50 else stats.normaltest(group1)
    _, p_norm2 = stats.shapiro(group2) if n2 < 50 else stats.normaltest(group2)
    normal = (p_norm1 > alpha) and (p_norm2 > alpha)
    results['normalité'] = {'groupe1': round(p_norm1, 4), 'groupe2': round(p_norm2, 4), 'ok': normal}
    
    # --- Étape 2 : Égalité des variances (si normal) ---
    if normal:
        _, p_var = stats.levene(group1, group2)
        equal_var = p_var > alpha
        results['égalité_variances'] = {'p_levene': round(p_var, 4), 'ok': equal_var}
    
    # --- Étape 3 : Test ---
    if normal:
        test_label = f"t-test {'Student' if equal_var else 'Welch'}"
        stat, p = stats.ttest_ind(group1, group2, equal_var=equal_var)
    else:
        test_label = "Mann-Whitney U (non paramétrique)"
        stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    results['test'] = test_label
    results['statistique'] = round(float(stat), 4)
    results['p_value'] = round(float(p), 4)
    results['décision'] = 'Rejet H₀' if p < alpha else 'Pas de rejet H₀'
    
    # --- Étape 4 : Taille d'effet ---
    n_total = n1 + n2
    s_pooled = np.sqrt(
        ((n1-1)*np.std(group1, ddof=1)**2 + (n2-1)*np.std(group2, ddof=1)**2) / (n1+n2-2)
    )
    d = (np.mean(group1) - np.mean(group2)) / s_pooled
    results["cohen_d"] = round(d, 3)
    results["interprétation_d"] = (
        "négligeable" if abs(d) < 0.2 else
        "petit"       if abs(d) < 0.5 else
        "moyen"       if abs(d) < 0.8 else "grand"
    )
    
    return results

# Exemple d'utilisation
np.random.seed(42)
A = np.random.normal(0.90, 0.03, 25)
B = np.random.normal(0.87, 0.03, 25)

rapport = full_test_pipeline(A, B)
for k, v in rapport.items():
    print(f"{k:25s} : {v}")
```

---

### 7.2 Calcul de la puissance et taille d'échantillon requise

```python
from statsmodels.stats.power import TTestIndPower

analysis = TTestIndPower()

# Combien de sujets pour détecter un effet moyen (d=0.5) avec 80% de puissance ?
n_required = analysis.solve_power(
    effect_size=0.5,   # Cohen's d attendu
    power=0.80,        # puissance souhaitée (1 - β)
    alpha=0.05,        # seuil (risque de type I)
    ratio=1.0          # taille des groupes identique
)
print(f"Taille d'échantillon requise par groupe : {n_required:.0f}")

# Quelle est la puissance avec n=20 et d=0.5 ?
power = analysis.power(effect_size=0.5, nobs1=20, alpha=0.05)
print(f"Puissance avec n=20 : {power:.1%}")
```

---

### 7.3 Visualisation complète des résultats

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def visualize_test(group1, group2, labels=('Groupe A', 'Groupe B'), alpha=0.05):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # --- Histogrammes avec KDE ---
    ax = axes[0]
    for data, label, color in zip([group1, group2], labels, ['steelblue', 'tomato']):
        ax.hist(data, bins=10, alpha=0.5, color=color, label=label, density=True)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(min(data)-1, max(data)+1, 200)
        ax.plot(x, stats.norm.pdf(x, np.mean(data), np.std(data, ddof=1)),
                color=color, linewidth=2)
    ax.legend()
    ax.set_title('Distributions')
    
    # --- Boxplots ---
    ax = axes[1]
    ax.boxplot([group1, group2], labels=labels, patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax.set_title('Boîtes à moustaches')
    
    # --- QQ-plots (normalité) ---
    ax = axes[2]
    for data, color, label in zip([group1, group2], ['steelblue', 'tomato'], labels):
        (osm, osr), (slope, intercept, r) = stats.probplot(data)
        ax.scatter(osm, osr, color=color, alpha=0.6, s=30, label=label)
    x_line = np.linspace(-3, 3, 100)
    ax.plot(x_line, x_line, 'k--', linewidth=1, alpha=0.5)
    ax.legend()
    ax.set_title('QQ-plot (normalité)')
    ax.set_xlabel('Quantiles théoriques')
    ax.set_ylabel('Quantiles observés')
    
    plt.tight_layout()
    plt.show()

# Utilisation
np.random.seed(42)
A = np.random.normal(0.90, 0.03, 25)
B = np.random.normal(0.87, 0.03, 25)
visualize_test(A, B, labels=('Modèle A', 'Modèle B'))
```

---

## 8. Aide-mémoire — quel test choisir ?

```
Données continues ?
├── 1 groupe vs valeur de référence       → t-test à 1 échantillon
├── 2 groupes indépendants
│   ├── Données normales                  → t-test de Welch
│   └── Données non normales             → Mann-Whitney U
├── 2 mesures sur les mêmes sujets        → t-test apparié (ou Wilcoxon)
└── 3+ groupes
    ├── Données normales, variances égales → ANOVA à 1 facteur
    │   └── Si significatif               → Post-hoc Tukey / Bonferroni
    └── Données non normales              → Kruskal-Wallis

Données catégorielles ?
├── 1 variable, distribution attendue     → Chi-deux de conformité (goodness-of-fit)
└── 2 variables, indépendance             → Chi-deux d'indépendance
    └── 2×2 avec petits effectifs         → Test exact de Fisher

Relation entre 2 variables continues ?
├── Relation linéaire                     → Pearson + régression linéaire
└── Relation monotone (non linéaire)      → Spearman

Normalité d'un échantillon ?
├── n < 50                                → Shapiro-Wilk
└── n ≥ 50                                → D'Agostino-Pearson
```

---

### Récapitulatif des fonctions Python

| Test | Fonction scipy | Résultat |
|------|---------------|---------|
| t-test 1 échantillon | `stats.ttest_1samp(data, popmean)` | `(t, p)` |
| t-test 2 échantillons | `stats.ttest_ind(a, b, equal_var=False)` | `(t, p)` |
| t-test apparié | `stats.ttest_rel(before, after)` | `(t, p)` |
| ANOVA | `stats.f_oneway(g1, g2, g3)` | `(F, p)` |
| Chi-deux | `stats.chi2_contingency(table)` | `(χ², p, ddl, expected)` |
| Mann-Whitney | `stats.mannwhitneyu(a, b)` | `(U, p)` |
| Kruskal-Wallis | `stats.kruskal(g1, g2, g3)` | `(H, p)` |
| Shapiro-Wilk | `stats.shapiro(data)` | `(W, p)` |
| Levene | `stats.levene(a, b)` | `(W, p)` |
| Pearson | `stats.pearsonr(x, y)` | `(r, p)` |
| Spearman | `stats.spearmanr(x, y)` | `(rho, p)` |

---

> **Note importante pour les praticiens ML/DL**
> 
> En apprentissage automatique, les datasets sont souvent très grands. Avec n = 100 000 observations, presque n'importe quelle différence devient statistiquement significative — même celle qui n'a aucun intérêt pratique. Dans ce contexte, la taille d'effet et les intervalles de confiance sont encore plus importants que la p-value. Utilisez les tests statistiques comme des outils de décision, pas comme des oracles.

---

*Document rédigé pour un cours destiné aux étudiants en ML/DL ayant des notions en Python.*
