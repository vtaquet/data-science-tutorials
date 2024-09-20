# 1. Estimation de l'incertitude résiduelle par estimation du Maximum de Vraisemblance: theorie

## Fonction de masse et de variable aléatoire à densité


Toute distribution de probabilité a soit une fonction de masse (si la distribution est discrète) ou une variable aléatoire à densité (si la distribution est continue). Cette fonction indique la probabilité qu'un échantillon prenne une valeur particulière. Dénotons cette fonction $`P(y | \theta)`$ où $`y`$ est la valeur de notre échantillon et $`\theta`$ est le paramètre qui décrit la distribution de probabilité:

```math
P(y | \theta) = \text{Prob} (\text{sampling value $`y`$ from a distribution with parameter $`\theta`$}).
```

Lorsque plusieurs échantillons sont tirés *indépendamment* de la même distribution (ce qui est une hypothèse courante), la fonction de masse ou de densité des valeurs échantillonnées $`y_1, \ldots, y_n`$ est le produit des fonctions de masse ou de densité de chaque $`y_i`$ individuel, ce qui donne:

```math
P(y_1, \ldots, y_n | \theta) = \prod_{i=1}^n P(y_i | \theta).
```


## La fonction de vraisemblance

Les variables aléatoires à densité sont habituellement considérées comme des fonctions de $`y_1, \ldots, y_n`$, avec le paramètre $`\theta`$ considéré comme fixe. Elles sont ainsi utilisées lorsqu'on connait le paramètre $`\theta`$ et que l'on souhaite obtenir la probabilité d'un échantillon prenant les valeurs $`y_1, \ldots, y_n`$. Cette fonction est couramment utilisée en *probabilité* lorsque la distribution est connue et qu'on souhaite déduire les valeurs échantillonnées de cette distribution.

A l'inverse, pour la fonction de *vraisemblance* les valeurs $`y_1, \ldots, y_n`$ sont considérées comme fixes et $`\theta`$ est la variable indépendante. Cette fonction est ainsi utilisée lorsque les valeurs échantillonnées $`y_1, \ldots, y_n`$ sont connues (à partir de données collectées) avec le paramètre $`\theta`$ encore inconnu. Cette fonction est ainsi utilisée en *statistiques* lorsque les données sont connues et que l'on souhaite effectuer des inférences sur la distribution originelle.

Ainsi, $`P(y_1, \ldots, y_n | \theta)`$ est appelée la *variable aléatoire à densité* lorsqu'elle fonction de $`y_1, \ldots, y_n`$ et que $`\theta`$ est fixe. Elle est appelée *vraisemblance* lorsqu'elle est fonction de $`\theta`$ avec $`y_1, \ldots, y_n`$ fixées et est dénotée $`L`$ où

```math
\underbrace{L(y_1, \ldots, y_n | \theta)}_{\text{ likelihood,} \\ \text{function of $`\theta`$}} = \underbrace{P(y_1, \ldots, y_n | \theta)}_{\text{probabiliy mass/density,} \\ \text{ function of $`y_1, \ldots, y_n`$}}
```


### Distribution de Bernoulli

La [distribution de Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution) est la distribution qu'une variable aléatoire prenne la valeur égale à 1 avec une probabilité $`\theta`$ et égale à 0 avec une probabilité de $`1-\theta`$. Soit $`P(y | \theta)`$ la probabilité qu'un événement retourne la valeur $`y`$ donné un paramètre $`\theta`$. 

avec le paramètre $`\theta`$. It's the distribution of a random variable that takes value 1 with probability $`\theta`$ and 0 with probability $`1-\theta`$. Let $`P(y | \theta)`$ be the probability that the event returns value $`y`$ given parameter $`\theta`$, alors

```math
\begin{align}
L(y | \theta) = P(y | \theta) &= \begin{cases}
1 - \theta \quad \text{if} \, y = 0 \\
\theta \quad \quad \, \, \, \text{if} \, y = 1 \\
\end{cases} \\
&= (1 - \theta)^{1 - y} \theta^y \quad y \in \{0, 1\}
\end{align}
```

Si les échantillons sont indépendants, alors nous obtenons
```math
L(y_1, \ldots, y_n | \theta) = \prod_{i=1}^n (1 - \theta)^{1 - y_i} \theta^{y_i}.
```

Ainsi, la probabilité d'observer $`0, 0, 0, 1, 0`$ est

```math
L(0, 0, 0, 1, 0 | \theta) = (1 - \theta)(1 - \theta)(1 - \theta)\theta(1 - \theta) = \theta(1 - \theta)^4.
```

Ici, les données sont donc fixes et la fonction de *vraisemblance* dépend de $`\theta`$ comme le montre la Figure ci-dessous. 

![bernouilli_distrib](figures/bernoulli_likelihood.png)


### Distribution normale (ou gaussienne)

Cette idée se généralise également naturellement à la [distribution normale](https://en.wikipedia.org/wiki/Normal_distribution). Cette distribution a deux paramètres: la moyenne $`\mu`$ et l'écart type $`\sigma`$. Nous avons donc $`\theta = (\mu, \sigma)`$. La variable aléatoire à densité (ou pdf pour *probability density function*) est donc

```math
L(y | \theta) = P(y | \theta) = P(y | \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \Big( - \frac{1}{2 \sigma^2} (y - \mu)^2 \Big).
```

Pour une séquence d'observations indépendantes $`y_1, \ldots, y_n`$, la vraisemblance est

```math
L(y_1, \ldots, y_n | \mu, \sigma) = \prod_{i=1}^n \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \Big( - \frac{1}{2 \sigma^2} (y_i - \mu)^2 \Big).
```

La vraisemblance est donc identique, mais vue comme une fonction de $`\mu`$ et $`\sigma`$ avec $`y_1, \ldots, y_n`$ vus comme constants. Par exemple, si les données observéees sont -1, 0, 1, la vraisemblance devient 

```math
L(-1, 0, 1 | \mu, \sigma) = (2 \pi \sigma^2)^{-3/2} \exp \Big( - \frac{1}{2 \sigma^2} [ (\mu-1)^2 + (\mu)^2 + (\mu+1)^2 ] \Big).
```

que nous pouvons tracer comme une fonction de $`\mu`$ et $`\sigma`$ comme suit

![normal_distrib](figures/gaussian_likelihood.png)

## Estimation du maximum de vraisemblance

La fonction de vraisemblance est communément utilisée en inférence statistique pour *fitter* une distribution à des données comme suit. Supposons que nous avons observé des données $`y_1, \ldots, y_n`$ qui suivent une distribution avec un paramètre inconnu $`\theta`$ que nous voulons estimer. La vraisemblance est donc

```math
L(y_1, \ldots, y_n | \theta).
```

L'estimation du maximum de vraisemblance $`\theta_{\text{MLE}}`$ (pour *Maximum Likelihood Estimate*) est ainsi la valeur qui maximise la vraisemblance $`L(y_1, \ldots, y_n | \theta)`$. Pour l'exemple précédent de la distribution de Bernoulli avec les données observées 0, 0, 0, 1, 0, cela nous donne $`p=\frac{1}{5}`$, qui est la valeur maximale de la courbe ci-dessus. Pour la distribution normale avec les données observées -1, 0, 1, le maximum de vraisemblance est donné par la zone autour de $`\mu=0, \sigma=\sqrt{\frac{2}{3}}`$. Ainsi, nous cherchons *les valeurs de paramètres qui rendent les données observées les plus plausibles*. Mathématiquement,  

```math
\theta_{\text{MLE}} = \arg \max_{\theta} L(y_1, \ldots, y_n | \theta).
```


### La log-vraisemblance négative


Pour des observations indépendantes, la vraisemblance devient un produit

```math
L(y_1, \ldots, y_n | \theta) = \prod_{i=1}^n L(y_i | \theta).
```

La fonction $`\log`$ étant strictement croissante, maximiser la vraisemblance revient à maximiser la log-vraisemblance $`\log L(y_1, \ldots, y_n | \theta)`$. Le produit ci-dessus devient donc une somme:

```math
\begin{align}
\theta_{\text{MLE}} &= \arg \max_{\theta} L(y_1, \ldots, y_n | \theta) \\
&= \arg \max_{\theta} \log L(y_1, \ldots, y_n | \theta) \\
&= \arg \max_{\theta} \log \prod_{i=1}^n L(y_i | \theta) \\
\theta_{\text{MLE}} &= \arg \max_{\theta} \sum_{i=1}^n \log L(y_i | \theta) \\
\theta_{\text{MLE}} &= \arg \min_{\theta} \sum_{i=1}^n - \log L(y_i | \theta).
\end{align}
```

La convention en optimisation est de toujours minimiser une fonction au lieu de la maximiser. Ainsi, maximiser la vraisemblance revient à minimiser la log-vraisemblance négative (ou NLL pour *Negative Log-Likelihood*):

```math
\theta_{\text{MLE}} = \arg \min_{\theta} \text{NLL}(y_1, \ldots, y_n | \theta)
```

où la NLL est définie comme

```math
\text{NLL}(y_1, \ldots, y_n | \theta) = - \sum_{i=1}^n \log L(y_i | \theta).
```

## Entrainement des réseaux de neurones

Cette méthode peut s'appliquer pour entrainer les réseaux neurones en choisissant les poids du réseau qui maximisent la vraisemblance (ou plutôt qui minimise la log-vraisemblance négative) d'observer des données d'entrainement. Le réseau de neurones peut être vue comme une fonction qui attribue un point $`x_i`$ au paramètre $`\theta`$ d'une distribution. Ce paramètre indique la probabilité de voir chaque label possible. Les vrais labels et la vraisemblance sont ainsi utilisés pour trouver les meilleurs poids du réseau de neurone.

Soit un réseau de neurones $`\text{NN}`$ avec des poids $`\mathbf{w}`$. Soit $`x_i`$ un point des données, par exemple une image à classifier ou un vecteur $`x`$ pour lequel nous voulons prédire la valeur $`y`$. La prédiction du réseau de neurones $`\hat{y}_i`$ est 

```math
\hat{y}_i = \text{NN}(x_i | \mathbf{w}).
```

Nous pouvons ainsi entrainer le réseau de neurones (et déterminer ses poids $`\mathbf{w}`$) comme suit. Supposons qu'une prédiction du réseau de neurones $`\hat{y}_i`$

Supposons que la prédiction de réseau de neurones $`\hat{y}_i`$ fait partie d'une distribution à partir de laquelle le vrai label est tiré. Supposons que nous ayons des données d'entraînement constituées d'entrées et des labels associés. Soit les données $`x_i`$ et les labels $`y_i`$ pour $`i=1, \ldots, n`$, où $`n`$ est le nombre d'échantillons d'entrainement. Les données d'entraînement sont donc

```math
\text{training data} = \{(x_1, y_1), \ldots, (x_n, y_n)\}
```

Pour chaque point $`x_i`$, nous avons la prédiction du réseau $`\hat{y}_i = \text{NN}(x_i | \mathbf{w})`$, qui spécifie une distribution. Nous avons également le vrai label $`y_i`$. Les poids du réseau de neurones entrainé sont donc ceux qui minimisent la log-vraisemblance négative:

```math
\begin{align}
\mathbf{w}^* &= \arg \min_{\mathbf{w}} \big( - \sum_{i=1}^n \log L(y_i | \hat{y}_i) \big) \\
&= \arg \min_{\mathbf{w}} \big( - \sum_{i=1}^n \log L(y_i | \text{NN}(x_i | \mathbf{w})) \big)
\end{align}
```

En pratique, déterminer le vrai optimum $`\mathbf{w}^*`$ n'est pas toujours possible. A la place, une valeur approximée est recherchée avec une descente de gradient stochastique, habituellement via une *backpropagation* des dérivées et un algorithme d'optimisation tel que `RMSprop` ou `Adam`.

L'annexe vous présente un exemple plus concret de régression montrant que la minimisation de la log-vraisemblance négative est équivalente à une régression des moindres carrés lorsqu'on supporte un terme d'erreur Gaussien de variance constante.
