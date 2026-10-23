CoachEntretien IA — agent conversationnel de préparation aux entretiens IA / Data Science
Mini-projet 2 : du Transformer pré-entraîné à une application de dialogue
==========================================================================================

1. CONTENU DU RENDU
-------------------
Dépôt GitHub : https://github.com/Charf91/Agent_Conversationnel_Entretien

agent.ipynb        Notebook exécuté, sections 1 à 7 :
                     1  Définition de l'agent (rôle, périmètre, 5 conversations types)
                     2  Observation du modèle (tokenisation, logits, softmax, décodage)
                     3  Baseline : comparaison de 3 prompts (A, B, C) et historique borné
                     4  Génération et contrôle des données
                     5  Adaptation LoRA
                     6  Évaluation avant / après sur 20 requêtes de test
                     7  Interface Gradio
demo_adapter.ipynb Utiliser l'agent sans réentraîner : télécharge le modèle de base et
                   l'adaptateur entraîné, puis interface Gradio de comparaison base / LoRA
rapport.pdf        Rapport (4 pages)
README.txt         Ce fichier
data/              train.jsonl (389), val.jsonl (48), test.jsonl (48) dialogues
results/           sorties de l'exécution (voir section 5)

Adaptateur LoRA (release GitHub v1.0, 32 Mo) :
https://github.com/Charf91/Agent_Conversationnel_Entretien/releases/download/v1.0/coach_lora_adapter.zip
Contenu : adapter_model.safetensors (matrices LoRA), adapter_config.json,
training_summary.json (hyperparamètres et meilleur pas).

Les données sont générées par le notebook (section 4, graine fixe 2026) : relancer le
notebook reproduit exactement les mêmes fichiers data/.

2. RÉSULTATS PRINCIPAUX
-----------------------
Section 3  Prompt retenu : C (consigne concise + 6 exemples), moyenne 1,10 / 2
           contre 0,83 (A, consigne détaillée) et 0,78 (B, consigne concise).
Section 5  8,8 M paramètres entraînés sur 502,8 M (1,75 %) ; perte de validation
           1,31 -> 0,1685 ; arrêt au maximum de 5 époques (245 pas).
Section 6  Perte sur data/test.jsonl : 2,32 (base) -> 0,21 (LoRA).
           Grille humaine (0 à 2)   base   LoRA
             P pertinence           0,80   0,80
             R rôle et format       0,85   1,70
             H historique           1,40   1,40
             D hors périmètre       1,70   1,80
             I non-invention        1,55   1,25
           LoRA apprend le rôle, le format et les redirections, mais pas de
           connaissances nouvelles : sur des notions absentes de l'entraînement,
           ses feedbacks sont bien structurés mais faux.

3. PRÉREQUIS
------------
- Un compte Google (Google Colab) ; Google Sheets pour les notations.
- Un GPU : Colab gratuit avec un GPU T4 suffit.
- Aucune clé Hugging Face n'est nécessaire (modèle public). L'avertissement
  « unauthenticated requests to the HF Hub » peut être ignoré.
- Dépendances (installées par le notebook si absentes) : torch, transformers (5.x),
  accelerate, peft, gradio, pandas, numpy, matplotlib. Le notebook désinstalle
  torchao, préinstallé dans Colab et incompatible avec peft.

4. REPRODUIRE L'EXÉCUTION (environ 1 h, dont 30 min de calcul)
--------------------------------------------------------------
Étape 1 — Ouvrir
  colab.research.google.com -> Importer -> agent.ipynb
  Exécution -> Modifier le type d'exécution -> T4 GPU -> Enregistrer
  Modifier -> Effacer toutes les sorties

Étape 2 — Sections 1 à 3
  Cliquer sur la cellule de titre « Sections 4 à 7 »,
  puis Exécution -> Exécuter les cellules précédentes (5 à 10 min).
  Vérifier : « Device : cuda », « GPU : Tesla T4 », 36 réponses de comparaison.

Étape 3 — Notation des prompts (section 3)
  Téléverser dans results/ le fichier results/manual_evaluation_template_v2.csv du dépôt
  (déjà noté), puis relancer les deux dernières cellules de la section 3.
  Résultat attendu : « Configuration provisoirement retenue : C_few_shot ».
  Remarque : les réponses générées peuvent varier légèrement d'une exécution à l'autre ;
  pour une nouvelle exécution, renoter de 0 à 2 dans Google Sheets.

Étape 4 — Sections 4 à 7
  Cliquer sur la cellule « Sections 4 à 7 »,
  puis Exécution -> Exécuter la cellule sélectionnée et les suivantes (20 à 25 min).

Étape 5 — Notation du test final (section 6)
  Téléverser dans results/ le fichier results/final_manual_scores.csv du dépôt (déjà noté :
  P, R, D, I de 0 à 2, H pour les id 15 à 19), puis relancer la cellule de notation.

Étape 6 — Sauvegarder
  Télécharger les dossiers data/ et results/, coach_lora_adapter.zip, puis
  Fichier -> Télécharger -> Télécharger le fichier .ipynb.
  Colab efface les fichiers à la déconnexion : télécharger dès la fin de l'exécution.

5. FICHIERS DANS results/
-------------------------
prompt_comparison_v2.json          réponses des prompts A, B, C (section 3)
manual_evaluation_template_v2.csv  notation des 36 réponses de prompts (section 3)
prompt_fix_tests_v2.json           tests ciblés (section 3)
final_test_queries.json            20 requêtes du test final
final_comparison.json              réponses avant / après adaptation
final_manual_scores.csv            notation des 40 réponses du test final (section 6)
final_evaluation_summary.csv       moyennes par critère, base vs LoRA
loss_curves.png, training_log.json suivi de l'entraînement

6. UTILISER L'ADAPTATEUR SANS RÉENTRAÎNER
-----------------------------------------
Méthode simple : ouvrir demo_adapter.ipynb dans Colab (GPU T4 conseillé), puis
Exécution -> Tout exécuter. L'adaptateur est téléchargé automatiquement depuis la
release GitHub ; l'interface Gradio apparaît en 1 à 2 minutes, avec un sélecteur
« après entraînement (LoRA) / avant entraînement (modèle de base) ».

Code minimal équivalent :
!pip uninstall -y -q torchao && pip install -q peft
!wget -q https://github.com/Charf91/Agent_Conversationnel_Entretien/releases/download/v1.0/coach_lora_adapter.zip
!unzip -q coach_lora_adapter.zip
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok  = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype="float32")
model = PeftModel.from_pretrained(base, "coach_lora_adapter")   # modèle adapté
L'adaptateur ne contient que les matrices LoRA (8,8 M paramètres, ~35 Mo) : le modèle de
base est téléchargé automatiquement depuis Hugging Face.

7. PROBLÈMES FRÉQUENTS
----------------------
« Device : cpu »            GPU non activé : refaire l'étape 1, puis Exécution -> Redémarrer et tout exécuter.
ImportError torchao         Exécuter « !pip uninstall -y -q torchao », redémarrer la session, relancer.
CUDA out of memory          Exécution -> Redémarrer la session et relancer ; si besoin, passer
                            per_device_train_batch_size à 1 et gradient_accumulation_steps à 8.
Session déconnectée         Les fichiers sont perdus : relancer depuis le début (données et
                            test identiques grâce aux graines fixes).
CSV mal lu                  Ouvrir et exporter avec Google Sheets (Excel en français change
                            le séparateur en « ; »).

8. MODÈLE ET HYPERPARAMÈTRES
----------------------------
Base       Qwen/Qwen2.5-0.5B-Instruct, gelé, poids float32, sans quantification
LoRA       r=16, alpha=32, dropout=0.05, modules q/k/v/o/gate/up/down
Optim.     lr 2e-4, planning cosinus, 10 pas de warmup, lots de 2 x 4 (accumulation) = 8,
           précision mixte fp16
Durée      5 époques max, arrêt précoce (patience 2) sur la perte de validation ;
           meilleur adaptateur sauvegardé par un callback (poids LoRA uniquement)
Longueur   768 tokens maximum ; perte calculée sur la dernière réponse de l'assistant
