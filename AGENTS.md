# AGENTS.md — instructions pour toute IA qui travaille dans ce dépôt

Équipe **TercerHombre** · More than a Hack 2027 (Metrica Sports × Barça Innovation Hub × MWCapital)
Réponds en **français**. Le code, les commentaires et les messages de commit sont en **anglais**.

---

## 1. Mission

Phase 1 (Initial Challenge) : pour chacune des **199 319 images** de la vidéo tactique de
FC Barcelona – Athletic Club, écrire `H` (Barça en possession), `A` (Athletic), `D` (ballon mort)
ou `X` (abstention). Le classement se fait sur **spellF1**, avec le macroF1 pour départager.

* **spellF1** : chaque possession (suite maximale de H ou de A) est un objet. Les possessions
  prédites sont appariées une à une aux vraies par IoU temporel, et le F1 est moyenné sur les
  seuils 0,3 / 0,4 / 0,5 / 0,6 / 0,7. L'ennemi est le **scintillement** : un changement d'équipe
  parasite d'une demi-seconde détruit une possession. Une seconde de retard partout = 0,65.
* Pour spellF1, un `X` est remplacé par la dernière étiquette : gratuit au milieu d'une
  possession, coûteux à un changement de possession.
* La définition de référence est dans `evaluation/possession_metrics.py`. Si un document la
  contredit, le code a raison.

**Échéance : 17 novembre 2026.** Gel du code le 14, soumission le 16.
Objectif : top 20 au classement (étape suivante : Innovation Challenge, puis finale à Barcelone).

---

## 2. Règles non négociables

Une seule violation peut disqualifier l'équipe. En cas de doute : **s'arrêter et demander.**

1. **Ne jamais lire, afficher ni transmettre les données du match.** Les conditions d'utilisation
   interdisent de fournir les vidéos, les données ou des extraits à un LLM hébergé. Tu es un LLM
   hébergé. Donc :
   * n'ouvre **jamais** les fichiers de `data/`, `cache/` (dont `cache/*.npz` et
     `cache/viewer_pack.bin`), ni aucune sortie image par image dérivée du tracking ;
   * n'écris pas de script qui imprime des valeurs image par image du tracking. Seules les
     **sorties agrégées** sont autorisées : scores, comptes, moyennes, histogrammes, comme
     `python -m possession.diagnose` ;
   * pour comprendre le match, demande à l'humain de regarder dans `tools/viewer.html` et de
     décrire ce qu'il voit.
   Sont autorisés : le code, la vérité terrain publique `evaluation/gt/*_public.csv` (publique), nos
   propres fichiers de soumission `submissions/*.csv`, et les données ouvertes d'autres matchs
   (`validation/external_data/`, Metrica sample-data).
2. **Aucune étiquette humaine sur le match fourni**, en aucune quantité et pour aucun usage, pas
   même pour valider. Pas de règle ni de constante liée à des images précises du match
   (« entre l'image X et Y, mettre D » est interdit). Les étiquettes sortent d'un algorithme.
3. **Ne jamais ajuster sur les 5 minutes publiques.** Elles servent à vérifier que le pipeline
   est correct (format, alignement, sens des étiquettes), jamais à choisir entre deux versions
   ni à régler un paramètre. Elles ne contiennent que 17 possessions : c'est du bruit.
4. **Aucune donnée humaine sur ce match**, quelle que soit la source (Opta, StatsBomb, Wyscout…).
   Les données d'autres matchs sont autorisées, à condition de les citer dans le document technique.
5. **Reproductibilité exacte** : `python -m possession.pipeline --team TercerHombre` doit
   régénérer le même fichier, octet pour octet. Pas d'aléatoire sans graine fixe, pas d'appel
   réseau, pas de pickle.
6. Le dépôt est **public** : ne jamais committer `data/`, `cache/`, `submissions/`,
   `validation/external_data/` ni `.venv/` (vérifier `git status` avant chaque commit).

---

## 3. Architecture

| chemin | rôle |
|---|---|
| `possession/atd.py` | lecture du tracking brut (ATD) → tableaux en mètres, cache |
| `possession/features.py` | indices par image : distances au ballon, contrôle, touches de balle (Link & Hoernig 2017, ELASTIC 2025), ballon posé, activité |
| `possession/model.py` | caractéristiques fenêtrées + modèles d'émission (arbres boostés exportés en JSON, prédits en numpy) |
| `possession/emissions.py` | log-odds P(mort), P(domicile), relances issues du Smart Tagging (frappe détectée par position) |
| `possession/decode.py` | Viterbi sur {H, A, D} avec coûts de changement |
| `possession/pipeline.py` | point d'entrée, écrit `submissions/<team>.csv` ; `--check` lance le validateur et la vérification publique |
| `possession/weights.json`, `weights_trees.json` | modèle courant (v5) + coûts du décodeur |
| `possession/diagnose.py` | statistiques **agrégées** par étiquette (partageables) |
| `validation/external.py` | banc d'essai : 3 matchs Metrica + 7 matchs Bundesliga ouverts (DFL BallStatus/BallPossession), tracking dégradé pour imiter l'ATD |
| `validation/train.py` | entraînement + choix des coûts en validation croisée « un match exclu » sur spellF1 |
| `validation/fetch_idsse.py` | téléchargement des matchs Bundesliga (CC BY 4.0) |
| `tools/viewer.html`, `tools/export_viewer.py` | visualiseur local vidéo + timeline (à l'usage de l'humain) |
| `docs/TECHNICAL.md` | document technique à remettre : le tenir à jour à chaque changement |

Contraintes techniques :
* L'inférence n'utilise **que numpy et pandas**, sur Python ≥ 3.8 (l'humain a aussi un `.venv` en 3.13).
* scikit-learn ne sert qu'à l'entraînement (Python ≥ 3.11) ; les modèles sont exportés en JSON.

---

## 4. Comment mesurer un progrès

Le seul juge interne est le **banc d'essai** (`validation/train.py`) : 10 autres matchs, chacun
évalué sans avoir servi à l'entraînement.

Prérequis : Python ≥ 3.11 avec `pip install -r requirements.txt -r validation/requirements.txt`,
et deux variables d'environnement (indispensables dans un worktree, où `data/` et
`validation/external_data/` n'existent pas) :
`MTAH_SAMPLES` (clone de metrica-sports/sample-data), `MTAH_EXTERNAL_DIR` (matchs Bundesliga),
et `MTAH_DATA_DIR` (données du match, pour le pipeline).

```bash
# comparer une modification en quelques minutes (coûts actuels, rien n'est écrit)
python -m validation.train --team-variant touch --learner gbdt --quick
# réentraînement complet + recherche des coûts (~30-40 min), régénère possession/weights*.json
python -m validation.train --team-variant touch --learner gbdt
```

Sous-agents Cursor disponibles (`.cursor/agents/`) : `/evaluateur`, `/auditeur-regles`,
`/verificateur-livraison`, `/chercheur`, `/redacteur-doc`. Fiches de mission : `docs/AGENT_TASKS.md`.

* Rapporte toujours **avant / après** : spellF1 moyen en matchs exclus, nombre de possessions
  prédites contre vraies, délai médian des changements, part des possessions courtes retrouvées.
* Un écart de moins de ~0,01 sur la moyenne des 10 matchs n'est pas significatif ; s'il est
  faible, vérifie avec une 2e graine de dégradation (`--seeds 2`).
* On ne garde un changement que s'il améliore le banc **et** qu'il a une justification
  footballistique ou physique. Le chiffre public de `--check` n'est qu'une vérification.
* Toute constante du simulateur (`validation/external.py`) se calibre sur des statistiques
  **agrégées** du vrai match (`possession.diagnose`, groupées par nos prédictions) ou sur les
  autres matchs, jamais sur la vérité publique.
* Avant de livrer : `python evaluation/validate_submission.py submissions/TercerHombre.csv`, et
  deux exécutions du pipeline doivent donner le même hash.

---

## 5. État actuel (v5)

| banc v5, 10 matchs exclus | spellF1 |
|---|---|
| baseline officielle | 0,283 (0,2433 sur le vrai match : ce banc est un peu plus facile) |
| v5 | **0,641**, possessions courtes retrouvées 28 %, délai médian −0,04 s |

5 minutes publiques (v4, indicatif) : spellF1 0,537, 18 possessions prédites pour 17 vraies.

Faiblesses connues, par gain attendu :
1. **Possessions courtes** (< 3 s, ~30 % des possessions) : seulement 28 % retrouvées. Les
   pistes : coûts de transition dépendant de la durée (semi-Markov), ou un détecteur de
   récupération (touche adverse suivie d'une 2e touche de la même équipe).
2. **Ballons morts courts** (< 8 s) : souvent manqués, à cause du coût de transition élevé
   (`to_dead` / `restart` = 55).
3. Écart de nature entre étiquettes : DFL change de possession à chaque touche (fusion des
   passages adverses de moins de 1,6 s), Metrica raisonne en possession contrôlée.
4. Bruit de la balle dans le simulateur (`BALL_NOISE`) : 65 touches/min simulées contre ~43 dans l'ATD.

---

## 6. Façon de travailler

* Une hypothèse → une modification → une mesure sur le banc → une décision. Pas de grosses
  réécritures sans mesure.
* Explique chaque changement à l'humain en termes de football (entraîneur, bon programmeur) :
  quel défaut, quelle cause, quelle correction, quel effet mesuré.
* Mets à jour `docs/TECHNICAL.md` (approche, données utilisées, règles de possession, limites).
* Sois franc sur l'incertitude : ne présente jamais un chiffre du banc comme le score final.
