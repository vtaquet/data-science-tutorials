# 2. Estimation de l'incertitude épistémique par inférence variationnelle: théorie

## Introduction


Dans cette partie, nous cherchons à inférer l'incertitude des poids au sein des réseaux de neurones via la méthode appelée *Bayes by Backprop* introduite par [Blundell et al. (2015)](https://arxiv.org/pdf/1505.05424.pdf). L'approche déterministe classique consiste à inférer une valeur unique pour chaque poids. Cependant, ces valeurs sont sujets à une incertitude *épistémique* due à l'imperfection des données d'entrainement, qui né décrivent pas parfaitement la vraie distribution des données. 

L'incorporation de cette incertitude peut être effectuée par l'introduction d'une distribution de probabilité sur chaque poids, dont ses propriétés sont ensuite apprises ou *inférées*. Dans le cas d'une distribution normale, chaque poids $`w_i`$ du réseau de neurones est représenté par deux paramètres, la moyenne $`\mu_i`$ et l'écart type $`\sigma_i`$, dont les valeurs sont estimées par backpropagation. 

La Figure ci-dessous résume ainsi les deux cas:
- Réseau de neurones déterministe: $`w_i = \hat{w}_i`$
- Réseau de neurones avec incertitude des poids: $`w_i \sim N(\hat{\mu}_i, \hat{\sigma}_i)`$.

![image.png](https://i.stack.imgur.com/LFn1C.png)

L'incertitude des poids est ensuite propagée vers les paramètres de sortie et les prédictions du réseau de neurones. Une valeur de sortie, ou prédiction, est ainsi déterminée en deux étapes:
- Echantillonner chaque poids du réseau de neurones à partir de leur distribution respective inférée, donnant ainsi un jeu unique de poids.
- Utiliser ces poids pour déterminer une valeur de sortie $`\hat{y}_i`$


## Inférence Bayésienne

Par simplicité, nous considérons ici uniquement des distributions continues et la notation $`P`$ se réfère donc à une *densité de probabilité*. Dans le cas de distributions discrètes, $`P`$ représenterait une masse de probabilités et les intégrales seraient remplacées par des sommes, mais les formules resteraient les mêmes. 

Les méthodes Bayésiennes représentent un formalisme d'inférence statistique permettant de calculer la distribution d'un paramètre d'un modèle à partir de données. Dans le contexte des réseaux de neurones, nous cherchons à déterminer la distribution de poids (nos paramètres de modèle) à partir de données d'entraînement. Cette étape se base sur le théorème de Bayes:

```math
P(w | D) = \frac{P(D | w) P(w)}{\int P(D | w') P(w') \text{d}w'} 
```

avec les termes suivants:
- $`D`$ est notre jeu de données, e.g. $`x`$ et $`y`$ sont des paires de valeurs: $`D = \{(x_1, y_1), \ldots, (x_n, y_n)\}`$. $`P(D) = {\int P(D | w') P(w') \text{d}w'} = P(D)`$ est parfois appelée l'*evidence* (en français ?).
- $`w`$ est la valeur d'un poids.
- $`P(w)`$ est la *prior* ou *distribution a priori*, représentant notre croyance *a priori* sur la densité de probabilité d'un poids, i.e. la distribution supposée avant de voir les données.
- $`P(D | w)`$ est la *likelihood*, la *vraisemblance* d'observer les données $`D`$ sachant le poids $`w`$. 
- $`P(w | D)`$ est la densité a posteriori (ou *posterior*) de la densité de distribution d'un poids ayant la valeur $`w`$ sachant nos données d'entrainement. Elle est appelée *a posteriori* car elle représente la distribution de nos poids après avoir pris en compte les données d'entrainement. 

Remarque: le terme $`{\int P(D | w') P(w') \text{d}w'} = P(D)`$ ne dépend pas de $`w`$ ($`w`$ étant la variable d'intégration). C'est donc un terme de normalisation et le théorème de Bayes peut être simplifié en

```math
P(w | D) = \frac{P(D | w) P(w)}{P(D)}. 
```

En conclusion, le théorème de Bayes nous donne un moyen de combiner nos données avec une croyance a priori sur nos paramètres pour obtenir une distribution sur ces paramètres qui prennent en compte ces données. 


## Réseau de Neurones Bayésien: la théorie et la pratique

La formule précédente nous donne en théorie un moyen pour déterminer une distribution pour chaque poids de notre réseau de neurones:
1. Choisir une densité *a priori* $`P(w)`$.
2. Déterminer une vraisemblance $`P(D | w)`$ à partir des données d'entrainement $`D`$.
3. Déterminer une densité a posteriori $`P(w | D)`$ grâce au théorème de Bayes, qui est la distribution des poids. 

L'application pratique est plus délicate à implémenter car elle implique de résoudre ou d'approximer l'intégrale de la constante de la normalisation $`{\int P(D | w') P(w') \text{d}w'} = P(D)`$ qui est une intégrale difficile à calculer. Des méthodes d'approximation ont ainsi été proposées. 


## Méthode variationnelle Bayésienne

Les méthodes variationnelles Bayesiennes (ou *Variational Bayesian methods*) visent à approximer la *vraie* distribution a posteriori par une deuxième fonction, appelée posterior variationnelle. Cette fonction a une forme fonctionnelle connue, et évite ainsi de déterminer exactement le *vrai* posterior  $`P(w | D)`$. L'approximation d'une fonction par une autre ne va pas sans risque. Ainsi, une approximation trop simplifiée pourrait engendrer une distribution a posteriori inexacte. Ce posterior variationnelle a donc un certain nombre de paramètres, $`\theta`$, qui sont réglés afin d'approcher au mieux le vrai posterior.

Chaque poids du réseau a une densité, appelée posterior variationnelle, dénotée $`q(w | \theta)`$ (à la place de $`P(w | D)`$ pour le *vrai* posterior), paramétrée par $`\theta`$. Le but est d'approcher $`q(w | \theta)`$ à $`P(w | D)`$ autant que possible et donc de minimiser la "différence" entre $`q(w | \theta)`$ et $`P(w | D)`$. La différence entre ces deux distributions est souvent mesurée par la [divergence de Kullback-Leibler](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) $`D_{\text{KL}}`$ (appelée par la suite la divergence KL). Nous cherchons donc à déterminer $`\theta_{opt}`$ minimisant cette divergence. Le principe de la minimisation de la divergence KL peut être résumé par cette Figure (issue de Grave 2011):

![image.png](https://miro.medium.com/max/700/1*40KNMV9I4090DSTJ4O0oBg.gif)

La divergence KL entre deux distributions ayant respectivement des densités $`f(x)`$ et $`g(x)`$ est définie comme suit:
```math
D_{KL} (f(x) || g(x)) = \int f(x) \log \left( \frac{f(x)}{g(x)} \right) \text{d} x.
```
Nous utilisons ici la convention que $`\frac{0}{0} = 1`$, $`D_{KL}`$ est donc nulle lorsque $`f(x) \equiv g(x)`$.

En supposant les données $`D`$ comme constantes, la divergence KL entre $`q(w | \theta)`$ et $`P(w | D)`$ est donc:
```math
\begin{align}
D_{KL} (q(w | \theta) || P(w | D)) &= \int q(w | \theta) \log \left( \frac{q(w | \theta)}{P(w | D)} \right) \text{d} w \\
&= \int q(w | \theta) \log \left( \frac{q(w | \theta) P(D)}{P(D | w) P(w)} \right) \text{d} w \\
&= \int q(w | \theta) \log P(D) \text{d} w + \int q(w | \theta) \log \left( \frac{q(w | \theta)}{P(w)} \right) \text{d} w - \int q(w | \theta) \log P(D | w) \text{d} w \\
&= \log P(D) + D_{KL} ( q(w | \theta) || P(w) ) - \mathbb{E}_{q(w | \theta)}(\log P(D | w))
\end{align}
```

où:
- dans la dernière ligne $`\int q(w | \theta) \log P(D) \text{d}w = \log P(D) \int q(w | \theta) \text{d} w = \log P(D)`$ car $`q(w | \theta)`$ est une distribution de probabilité, son intégrale est donc égale à 1.
- $`\mathbb{E}_{q(w | \theta)}(\log P(D | w))`$ est l'espérance mathématique de la log-vraisemblance des données sous le posterior variationnel $`q(w | \theta)`$. Il est également appelé le *coût de vraisemblance* (likelihood cost) et dépend de $`\theta`$ et des données mais pas du prior. 

$`L(\theta | D)`$ est la fonction de perte à minimiser afin de déterminer le paramètre $`\theta`$. A partir de la dérivation précédente, nous obtenons

```math
\log P(D) = \mathbb{E}_{q(w | \theta)}(\log P(D | w)) - D_{KL} ( q(w | \theta) || P(w) ) + D_{KL} (q(w | \theta) || P(w | D))
```

$`D_{KL} (q(w | \theta) || P(w | D))`$ étant non-négatif, nous obtenons donc

```math
\log P(D) \ge \mathbb{E}_{q(w | \theta)}(\log P(D | w)) - D_{KL} ( q(w | \theta) || P(w) ) =: ELBO
```

Le terme de droite est donc une limite inférieure sur la log-vraisemblance, appelée ELBO pour *evidence lower bound*. L'ELBO est l'opposé de notre fonction de perte, minimiser la fonction de perte revient donc à maximiser cet ELBO.  

Maximiser l'ELBO nécessite un tradeoff entre le terme KL et le terme d'espérance de la log-vraisemblance. D'une part, la divergence entre $`q(w | \theta)`$ et $`P(w)`$ doit rester petite, de sorte que le posterior variationnel ne soit pas trop différence du prior. D'autre part, les paramètres du posterior variationnel doivent maximiser l'espérence de la log-vraisemblance $`\log P(D | w)`$, de sorte que le modèle attribue une vraisemlance plus élevée aux données.