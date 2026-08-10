# Analyse détaillée de `reward_function.py`

Ce fichier définit la fonction de récompense, qui est le **cœur de l'algorithme GRPO**. C'est elle qui note les phrases générées par le modèle (entre 0 et 1) pour qu'il puisse s'améliorer.

Contrairement à de nombreuses approches qui utilisent un deuxième gros modèle d'IA externe (comme BART ou un Reward Model) pour noter le texte, notre implémentation est **purement mathématique et auto-évaluée**. Elle utilise le modèle lui-même pour calculer la qualité du texte. C'est beaucoup plus rapide et efficace en mémoire !

Notre récompense totale est une somme pondérée de trois critères :
`Total Reward = (0.45 * CE) + (0.275 * SLOR) + (0.275 * Diversité)`

---

## 1. Control Effectiveness (CE) : Le respect de la consigne

**Score : 45% de la note finale**

Le CE (Efficacité du Contrôle) mesure si la phrase générée respecte bien l'étiquette demandée (Sentiment ou Topic).

```python
def control_effectiveness_score(prompt: str, completion: str, model, tokenizer):
    full_text = prompt + completion
    # On passe le texte entier dans le modèle Llama
    ...
    # On masque le prompt (labels = -100) pour ne calculer l'erreur (Loss) QUE sur la complétion
    labels[:, :prompt_len] = -100
    ...
    loss = outputs.loss.item()
    return math.exp(-loss)
```
- **Comment ça marche ?** On donne le texte entier (`prompt` + `completion`) à Llama. Le modèle calcule sa *Loss* (erreur) de Cross-Entropy (CE).
- En masquant le prompt (avec `-100`), on demande au modèle : *"Étant donné cette consigne, à quel point était-il logique/probable d'écrire cette réponse exacte ?"*
- Si la réponse correspond parfaitement à la consigne, la *Loss* (l'erreur) sera proche de 0.
- `math.exp(-loss)` transforme cette erreur (qui va de 0 à l'infini) en un score de probabilité parfait (entre 0.0 et 1.0).

---

## 2. SLOR (Fluency) : La qualité du langage

**Score : 27.5% de la note finale**

Le SLOR (Syntactic Log-Odds Ratio) vérifie si le texte généré ressemble à de l'anglais naturel, ou si le modèle a juste craché une bouillie de mots-clés.

```python
def fluency_score(completion: str, model, tokenizer) -> float:
    # On ne donne QUE la complétion au modèle (sans la consigne)
    ...
    loss = outputs.loss.item()
    return math.exp(-loss)
```
- **Comment ça marche ?** Cette fois, on ne donne **que** la réponse générée (`completion`) au modèle, sans aucun contexte.
- On regarde la *Loss* intrinsèque de cette phrase.
- Si le modèle a généré `"The food was great"`, la Loss sera très basse (c'est une phrase très commune).
- Si le modèle a généré `"great positive sports positive"`, la Loss va exploser car cette phrase n'a aucun sens linguistique.
- Encore une fois, `math.exp(-loss)` transforme l'erreur en un score entre 0.0 et 1.0.

---

## 3. Diversité (Distinct-N) : La créativité du modèle

**Score : 27.5% de la note finale**

Le gros risque du RL, c'est le "Mode Collapse" : le modèle trouve une phrase parfaite (ex: `"This is a great movie"`) et la répète en boucle à chaque itération parce qu'elle obtient une bonne note. La fonction de diversité empêche ça.

```python
def compute_distinct_n(text: str, n: int):
    tokens = text.strip().split()        
    ...
    # On groupe les mots par paquets de N (n-grams)
    ...
    return len(set(ngrams)) / len(ngrams)
```
- **Comment ça marche ?** On découpe la phrase en groupes de 1, 2, et 3 mots (les `n-grams`).
- On compare le nombre de groupes *uniques* par rapport au nombre total de groupes.
- Si le texte est `"food food food food"`, il n'y a qu'un seul mot unique, le score de diversité sera de 0.25 (très mauvais).
- Si le texte a un vocabulaire riche, le score se rapprochera de 1.0.

La fonction `diversity_score` calcule simplement la moyenne entre les Distinct-1, Distinct-2 et Distinct-3.

---

## Conclusion
```python
total_reward = Wce * r_ce + Wslor * r_slor + Wdiv * r_div
```
Pour chaque phrase, on fait la somme de ces trois scores pondérés. 
Cette approche mathématique est redoutable car elle force le modèle à trouver le juste équilibre : respecter la consigne (CE), tout en parlant un anglais parfait (SLOR), sans jamais se répéter bêtement (Diversité).
