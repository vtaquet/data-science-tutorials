# 2. Estimation de l'incertitude épistémique par inférence variationnelle: théorie (2e partie)

## La méthode de Backpropagation

Les idées présentées dans le notebook précédent peuvent être appliquées pour créer un réseau de neurones Bayésien ayant des incertitudes sur les poids. Supposons que nous voulons déterminer la distribution d'un poids particulier $`w`$:

1. Une distribution *a priori* avec une densité $`P(w)`$ est assignée au poids, qui représente notre croyance sur les valeurs possibles d'un réseau avant d'avoir vu les données d'entrainement. La distribution peut être simple, comme une distribution Normale, et n'a pas de paramètres "entraînables".

2. Un posterior variationnel avec une densité $`q(w | \theta)`$, où $`\theta`$ est le paramètre à inférer, est ensuite assigné au poids.

3. $`q(w | \theta)`$ est l'approximation de la vraie distribution posterieure du poids. $`\theta`$ doit donc être ajusté afin de rendre l'approximation la plus précise telle que mesurée par l'ELBO.

La question restante est alors de savoir comment déterminer $`\theta`$. Les réseaux de neurones sont généralement entraînés via un algorithme de backpropagation, dans lequel les poids sont mis à jour en les perturbant dans une direction qui réduit la fonction de perte. Nous cherchons à faire de même ici, en mettant à jour $`\theta`$ dans une direction qui réduit $`L(\theta | D)`$.

Ainsi, la fonction que nous cherchons à minimiser est

```math
\begin{align}
L(\theta | D) &= D_{KL} ( q(w | \theta) || P(w) ) - \mathbb{E}_{q(w | \theta)}(\log P(D | w)) \\
&= \int q(w | \theta) ( \log q(w | \theta) - \log P(D | w) - \log P(w) ) \text{d}w.
\end{align}
```

En principe, des dérivées de $`L(\theta | D)`$ par rapport à $`\theta`$ pourraient être calculées afin de mettre à jour la valeur de $`\theta`$. Cependant, cela implique de faire une intégrale sur $`w`$, un calcul en pratique quasi-impossible ou très lourd. Nous pourrions à la place écrire la fonction ci-dessus comme une espérance mathématique et ainsi utiliser une approximation de Monte Carlo pour calculer ses dérivées. À présent, nous pouvons écrire cette fonction sous la forme suivante

```math
\begin{align}
L(\theta | D) &= \mathbb{E}_{q(w | \theta)} ( \log q(w | \theta) - \log P(D | w) - \log P(w) ).
\end{align}
```

Cependant, prendre des dérivées en fonction de $`\theta`$ est compliqué car la distribution sous-jacente dépend de $`\theta`$. Une manière de contourner ce problème est par le *reparameterization trick*.

### Le *reparameterization trick*

Le *reparameterization trick* est un moyen de déplacer la dépendance sur $`\theta`$ afin qu'une espérance puisse être prise indépendamment d'elle. Voyons cela avec un exemple. Supposons que $`q(w | \theta)`$ est une Gaussienne, de sorte que $`\theta = (\mu, \sigma)`$. Alors, pour une fonction arbitraire $`f(w; \mu, \sigma)`$, nous avons

```math
\begin{align}
\mathbb{E}_{q(w | \mu, \sigma)} (f(w; \mu, \sigma) ) &= \int q(w | \mu, \sigma) f(w; \mu, \sigma) \text{d}w \\
&= \int \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( -\frac{1}{2 \sigma^2} (w - \mu)^2 \right) f(w; \mu, \sigma) \text{d}w \\
&= \int \frac{1}{\sqrt{2 \pi}} \exp \left( -\frac{1}{2} \epsilon^2 \right) f \left( \mu + \sigma \epsilon; \mu, \sigma \right) \text{d}\epsilon \\
&= \mathbb{E}_{\epsilon \sim N(0, 1)} (f \left( \mu + \sigma \epsilon; \mu, \sigma \right) )
\end{align}
```

où nous avons utilisé le changement de la variable $`w = \mu + \sigma \epsilon`$. Notez que la dépendance sur $`\theta = (\mu, \sigma)`$ n'est plus que dans l'intégrale et nous pouvons prendre les dérivées par rapport à $`\mu`$ et $`\sigma`$:

```math
\begin{align}
\frac{\partial}{\partial \mu} \mathbb{E}_{q(w | \mu, \sigma)} (f(w; \mu, \sigma) ) &= \frac{\partial}{\partial \mu} \mathbb{E}_{\epsilon \sim N(0, 1)} (f \left( w; \mu, \sigma \right) ) = \mathbb{E}_{\epsilon \sim N(0, 1)} \frac{\partial}{\partial \mu} f \left( \mu + \sigma \epsilon; \mu, \sigma \right)
\end{align}
```

```math
\begin{align}
\frac{\partial}{\partial \sigma} \mathbb{E}_{q(w | \mu, \sigma)} (f(w; \mu, \sigma) ) &= \frac{\partial}{\partial \sigma} \mathbb{E}_{\epsilon \sim N(0, 1)} (f \left( w; \mu, \sigma \right) ) = \mathbb{E}_{\epsilon \sim N(0, 1)} \frac{\partial}{\partial \sigma} f \left( \mu + \sigma \epsilon; \mu, \sigma \right)
\end{align}
```

Enfin, notons que nous pouvons approximer l'espérance par son estimation Monte Carlo:

```math
\begin{align}
\mathbb{E}_{\epsilon \sim N(0, 1)}  \frac{\partial}{\partial \theta} f \left( \mu + \sigma \epsilon; \mu, \sigma \right) \approx \sum_{i}  \frac{\partial}{\partial \theta} f \left( \mu + \sigma \epsilon_i; \mu, \sigma \right),\qquad \epsilon_i \sim N(0, 1).
\end{align}
```

Le *reparameterization trick* mentionné ici fonctionne dans les cas où nous pouvons écrire le $`w = g(\epsilon, \theta)`$, où la distribution de la variable aléatoire $`\epsilon`$ est indépendante de $`\theta`$.


### Implémentation

En mettant tous ces éléments ensemble, pour notre loss function $`L(\theta | D) \equiv L(\mu, \sigma | D)`$, nous obtenons

```math
f(w; \mu, \sigma) = \log q(w | \mu, \sigma) - \log P(D | w) - \log P(w)
```

```math
\begin{align}
\frac{\partial}{\partial \mu} L(\mu, \sigma | D) \approx \sum_{i} \left( \frac{\partial f(w_i; \mu, \sigma)}{\partial w_i} + \frac{\partial f(w_i; \mu, \sigma)}{\partial \mu} \right)
\end{align}
```

```math
\begin{align}
\frac{\partial}{\partial \sigma} L(\mu, \sigma | D) \approx \sum_{i} \left( \frac{\partial f(w_i; \mu, \sigma)}{\partial w_i} \sigma + \frac{\partial f(w_i; \mu, \sigma)}{\partial \sigma} \right)
\end{align}
```

```math
f(w; \mu, \sigma) = \log q(w | \mu, \sigma) - \log P(D | w) - \log P(w)
```

où $`w_i = \mu + \sigma \epsilon_i, \, \epsilon_i \sim N(0, 1)`$. En pratique, nous ne prenons souvent qu'un seul échantillon $`\epsilon_1`$ pour chaque point d'apprentissage. Cela conduit au plan de backpropagation suivant:
1. Échantillonner $`\epsilon_i \sim N(0, 1)`$.
2. Calculer $`w_i = \mu + \sigma \epsilon_i`$.
3. Calculer
```math
\nabla_{\mu}f = \frac{\partial f(w_i; \mu, \sigma)}{\partial w_i} + \frac{\partial f(w_i; \mu, \sigma)}{\partial \mu} \hspace{3em} \nabla_{\sigma}f = \frac{\partial f(w_i; \mu, \sigma)}{\partial w_i} \sigma + \frac{\partial f(w_i; \mu, \sigma)}{\partial \sigma}
```
4. Mettre à jour les paramètres avec un optimiseur basé sur le gradient en utilisant les gradients ci-dessus.

C'est comme ceci que les paramètres de la distribution sont appris pour chaque poids de réseau neuronal.


### Minibatches

Notons que notre fonction de perte (ou le négatif de l'ELBO) est
```math
\begin{align}
L(\theta | D) &= D_{KL} ( q(w | \theta) || P(w) ) - \mathbb{E}_{q(w | \theta)}(\log P(D | w)) \\
& = D_{KL} ( q(w | \theta) || P(w) ) - \sum_{j=1}^N \log P(y_j, x_j | w_j)
\end{align}
```

où $`j`$ parcourt tous les points des les données d'apprentissage ($`N`$ au total) et $`w_j = \mu + \sigma \epsilon_j`$ est échantillonné en utilisant $`\epsilon_j \sim N(0, 1)`$ (nous supposons un seul échantillon du posterior approximé par point de données pour plus de simplicité).

Si l'entraînement se produit dans des minibatchs de taille $`B`$, généralement beaucoup plus petits que $`N`$, nous avons à la place une fonction de perte

```math
\begin{align}
D_{KL} ( q(w | \theta) || P(w) ) - \sum_{j=1}^{B} \log P(y_j, x_j | w_j).
\end{align}
```

Notez que les facteurs de mise à l'échelle entre le premier et le deuxième terme ont changé, car la somme passait auparavant de 1 à $`N`$, mais elle passe maintenant de 1 à $`B`$. Pour corriger cela, nous devons ajouter un facteur de correction $`\frac{N}{B}`$ au deuxième terme pour nous assurer que son espérance est la même qu'auparavant. Après avoir divisé par $`N`$ pour prendre la moyenne par valeur d'entraînement, cela conduit à la fonction de perte

```math
\begin{align}
\frac{1}{N} D_{KL} ( q(w | \theta) || P(w) ) - \frac{1}{B} \sum_{j=1}^{B} \log P(y_j, x_j | w_j).
\end{align}
```

Par défaut, lorsque Tensorflow calcule la fonction de perte, il calcule la moyenne sur le minibatch. Par conséquent, il utilise déjà le facteur $`\frac{1}{B}`$ présent sur le deuxième terme. Cependant, il ne divise pas, par défaut, le premier terme par $`N`$. Dans une implémentation, nous devrons le préciser. 

## Conclusions

Ce notebook présente la méthode *Bayes by Backpropagation*, qui peut être utilisée pour incorporer l'incertitude des poids dans les réseaux de neurones. Cette approche permet de modéliser l'incertitude *épistémique* sur les poids du modèle. L'incertitude sur les poids du modèle est censée diminuer à mesure que le nombre de points d'entraînement augmente.
