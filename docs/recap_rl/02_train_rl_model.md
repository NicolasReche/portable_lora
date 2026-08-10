# Analyse détaillée de `train_rl_model.py`

Ce fichier est le script principal d'entraînement. Il configure l'environnement, charge le modèle Llama et lance la boucle d'apprentissage par renforcement (RL) avec l'algorithme GRPO.

---

## 1. La préparation des données GRPO

Contrairement à l'entraînement SFT où on donne la réponse complète (`prompt` + `completion`) pour forcer le modèle à la recopier, le GRPO a besoin que le modèle réfléchisse et génère sa propre réponse.

```python
def sample_to_prompt(sample: dict, attribute: str='sentiment'):
    prompt = f"[{attribute.upper()}] {sample['control']} [\\{attribute.upper()}] [ANS] {sample['input']}".strip()
    return {'prompt': prompt}
```
Cette fonction prend une ligne du dataset (un `sample`) et ne renvoie **que le prompt**. 
Le dataset de GRPO ne contiendra qu'une seule colonne : `prompt`. L'algorithme se chargera de donner ce prompt au modèle et de lui demander de générer le reste.

---

## 2. Le Chargement du Modèle (L'art de l'Adaptateur)

C'est ici que se joue toute l'architecture de votre projet (le "Portable LoRA").

```python
# 1. Chargement du modèle de base (Llama 3.1 8B) en 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    config['model']['base_model'],
    quantization_config=bnb_config,
    device_map="auto"
)
```
Pour éviter d'exploser la mémoire VRAM, on charge l'immense modèle Llama en version compressée (4-bit / NF4). Les poids de ce modèle de base **ne seront jamais modifiés** (ils sont "gelés").

```python
# 2. Chargement de l'Adaptateur SFT pour le RL
model = PeftModel.from_pretrained(
    model, 
    config['model']['sft_adapter_path'], 
    is_trainable=True
)
```
C'est l'étape cruciale : on superpose l'Adaptateur SFT (que vous avez entraîné précédemment) par-dessus le modèle de base gelé. 
- Pourquoi ? Parce qu'un modèle brut non-aligné ferait n'importe quoi en RL. Il faut partir d'un modèle qui sait déjà formater ses réponses (`[ANS]...[\ANS]`).
- L'argument `is_trainable=True` est fondamental : il indique à PyTorch que les poids de cet adaptateur SFT ont le droit d'être modifiés par l'algorithme RL. L'algorithme va donc faire "muter" l'adaptateur SFT pour le rendre meilleur.

---

## 3. Configuration de l'algorithme GRPO (`GRPOConfig`)

La classe `GRPOConfig` définit les règles du jeu pour l'entraînement RL.

```python
train_args = GRPOConfig(
    max_prompt_length=128,          # La taille max de la question
    max_completion_length=896,      # La taille max de la réponse générée
    num_generations=4,              # Le nombre de réponses à générer par prompt
    learning_rate=1e-5,             # Très petit en RL !
    optim="adamw_torch",            # L'optimiseur PyTorch stable
    ...
)
```
- **`num_generations = 4`** : C'est le principe même du **GRPO** (Group Relative Policy). Pour chaque prompt, Llama va générer un "groupe" de 4 réponses différentes. Le juge (BART) va donner une note à ces 4 réponses. Le GRPO va calculer la *moyenne* de ces 4 notes. Si la réponse n°1 a une note supérieure à la moyenne du groupe, l'algorithme modifiera les poids du modèle pour qu'il reproduise ce genre de réponse. Si la réponse n°3 a une note inférieure à la moyenne, l'algorithme "punira" le modèle.
- **`learning_rate = 1e-5`** : En RL, le taux d'apprentissage doit être beaucoup plus faible qu'en SFT, car l'entraînement est instable et basé sur des probabilités. Si on modifie trop fort les poids, le modèle s'effondre (il se met à répéter le même mot en boucle pour tromper le juge, ce qu'on appelle le "Reward Hacking").

---

## 4. Le GRPOTrainer

```python
trainer = GRPOTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    args=train_args,
    reward_funcs=[reward_function_v1], # La liste des juges
)

trainer.train()
```
Le `GRPOTrainer` encapsule toute la complexité mathématique. 
Dans les coulisses, pour chaque étape (`step`), il va :
1. Piocher des `prompts` dans le `train_dataset`.
2. Lancer la génération de texte avec Llama.
3. Envoyer le texte généré aux fonctions définies dans `reward_funcs`.
4. Récupérer les notes.
5. Calculer le Gradient (l'erreur) selon la formule PPO/GRPO (Relative Advantage).
6. Appliquer ce gradient via l'optimiseur (`adamw_torch`) pour mettre à jour les petits poids de l'Adaptateur LoRA.
