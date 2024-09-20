## Distribution Normale: régression des moindres carrés

L'idée présentée ci-dessus peut également s'appliquer pour un problème de régression. Nous avons une valeur $`x_i`$ et nous cherchons à prédire la valeur associée $`y_i`$ via un réseau de neurones qui donne une prédiction $`\hat{y}_i`$:

```math
\hat{y}_i = \text{NN}(x_i | \mathbf{w}).
```

Prenons l'exemple d'une régression linéaire avec les données suivantes.

![image.png](figures/linear_regression.png)

Il n'est pas possible de placer une droite passant par chaque point. De plus, même des points ayant des valeurs $`x`$ similaires peuvent ne pas avoir les mêmes valeurs de $`y`$. Nous pouvons ainsi interpréter ce problème comme $`y`$ étant linéairement relié à $`x`$ avec du bruit additionnel:

```math
y_i = f(x_i) + \epsilon_i \quad \quad  \epsilon_i \sim N(0, \sigma^2)
```

où $`f`$ est une fonction que nous cherchons déterminer et $`\epsilon_i`$ est du bruit Gaussien avec une moyenne de 0 et une variance $`\sigma^2`$. En deep learning, nous pouvons approximer $`f(x_i)`$ par un réseau de neurones $`\text{NN}(x_i | \mathbf{w})`$ ayant des poids $`\mathbf{w}`$ and des sorties $`\hat{y}_i`$.

```math
\hat{y}_i = \text{NN}(x_i | \mathbf{w}) = f(x_i)
```

Sous cette hypothèse, nous avons donc

```math
\epsilon_i = y_i - \hat{y}_i \sim N(0, \sigma^2)
```

et ainsi, donné des données d'entrainement $`\{(x_1, y_1), \ldots, (x_n, y_n)\}`$, nous obtenons la log-vraisemblance négative (en supposant l'indépendance des termes de bruit):

```math
\begin{align}
\text{NLL}((x_1, y_1), \ldots, (x_n, y_n) | \mathbf{w}) &= - \sum_{i=1}^n \log L(y_i | \hat{y}_i) \\
&= - \sum_{i=1}^n \log \Big( \frac{1}{\sqrt{2\pi\sigma^2}} \exp \Big( - \frac{1}{2\sigma^2} (\hat{y}_i - y_i)^2 \Big) \Big) \\
&= \frac{n}{2} \log (2\pi\sigma^2) + \frac{1}{2\sigma^2} \sum_{i=1}^n (\hat{y}_i - y_i)^2 \\
&= \frac{n}{2} \log (2\pi\sigma^2) + \frac{1}{2\sigma^2} \sum_{i=1}^n (\text{NN}(x_i | \mathbf{w}) - y_i)^2.
\end{align}
```

Puisque le dernier terme n'inclue que les poids, minimiser la log-vraisemblance négative est équivalent à minimiser

```math
\sum_{i=1}^n (\text{NN}(x_i | \mathbf{w}) - y_i)^2
```

ce qui revient à la somme des erreurs au carré. Ainsi, la régression des moindres carrés (où l'entrainement d'un réseau de neurones par la *mean squared error*) revient à entrainer un réseau de neurones pour reproduire la valeur attendue de sortie via la minimisation de la log-vraisemblance négative en supposant un terme d'erreur Gaussien de variance constante. 