# Fiches de mission pour les agents Cursor (semaine du 11 au 17 novembre)

Chaque fiche = **un agent dans son propre worktree** (Agents Window → nouvel agent → Worktree).
Un agent ne touche qu'aux fichiers de sa fiche. Toute amélioration se juge sur le banc
(`python -m validation.train --team-variant touch --learner gbdt --quick`) par le sous-agent
`/evaluateur`, puis passe `/auditeur-regles` avant d'être fusionnée.

Référence actuelle (v5) : banc 0,641 en matchs exclus · baseline 0,283 sur le même banc.

---

## A. Possessions courtes (gain attendu : élevé)

**Constat** : environ 30 % des possessions durent moins de 3 s ; on n'en retrouve que 28 %.
**Fichiers autorisés** : `possession/decode.py`, `possession/model.py`, `validation/train.py`.
**Pistes** :
1. Décodeur semi-Markov : un coût qui dépend de la durée de la possession, au lieu d'un coût
   fixe par changement.
2. Détecteur de récupération : une touche adverse suivie d'une 2e touche de la même équipe
   ouvre une possession, même brève.
3. Indices du modèle d'équipe propres aux duels : nombre d'adversaires à moins de 2 m,
   direction du ballon après la touche.

**Critère de réussite** : banc ≥ 0,66, et plus de possessions courtes retrouvées, sans
augmenter le nombre de fausses possessions.

**Prompt de lancement** :
```
Lis AGENTS.md puis docs/AGENT_TASKS.md, fiche A. Ne modifie que les fichiers autorisés.
1) Lance /evaluateur sur la base pour avoir la référence.
2) Propose-moi 2 approches (cause visée, changement, risque). Attends mon choix.
3) Implémente, relance /evaluateur, puis /auditeur-regles.
Termine par un bloc STATUT : READY / BLOCKED / STUCK + tableau avant/après.
```

## B. Arrêts de jeu courts et relances (gain attendu : moyen)

**Constat** : les arrêts de moins de 8 s sont souvent manqués (coûts `to_dead` / `restart` = 55).
**Fichiers autorisés** : `possession/emissions.py`, `possession/features.py`, `possession/model.py`.
**Pistes** : détecteur explicite « ballon perdu par la caméra près d'une ligne + joueurs qui
ralentissent » ; coûts différents pour entrer en ballon mort et pour en sortir ; durée
minimale d'un arrêt.

**Critère de réussite** : banc en hausse, et dans `python -m possession.diagnose`, les images
prédites mortes gardent un taux de « ballon vu » proche de 27 %.

**Prompt de lancement** : même modèle que A, avec « fiche B ».

## C. Mieux suivre le ballon depuis la vidéo (gain attendu : le plus élevé ; GPU disponible)

**Constat** : le tracking perd le ballon sur ~13 % des images de jeu, surtout dans les duels,
c'est-à-dire là où la possession change. Le règlement autorise explicitement à « run your own
detector, improve the tracking with your own algorithms ».

**Machine** : RTX 4070 Laptop (8 Go), Core Ultra 9 185H, 32 Go de RAM. Créer un environnement
dédié `.venv-vision`. Installer PyTorch avec CUDA via le sélecteur de pytorch.org ; si aucune
version n'existe pour Python 3.13 sous Windows, utiliser Python 3.12 pour cet environnement.
Vérifier que `torch.cuda.is_available()` vaut True avant toute autre chose.

**Idée clé : un détecteur entraîné sur NOTRE vidéo, sans aucune étiquette humaine.**
Le règlement autorise les « pseudo-labels produced by an algorithm, provided no person decided
any of them ». Le tracking automatique voit le ballon sur environ 57 % des images, et
l'homographie convertit ses positions en pixels. On obtient donc des dizaines de milliers
d'exemples « ballon ici » propres à cette caméra, sans qu'aucun humain n'ait annoté.

**Fichiers autorisés** : nouveau dossier `ballvision/` (+ `requirements-vision.txt`), et
`possession/atd.py` pour la fusion.

**Plan** :
1. `ballvision/pseudolabels.py` : pour les images où l'ATD voit le ballon, projeter sa position
   en pixels (`starter/load_homography.project`) et découper des imagettes centrées (positifs),
   plus des imagettes sans ballon (négatifs). Tout est écrit dans `cache/ballvision/`, jamais
   commité.
2. Choisir un modèle avec une licence compatible avec un dépôt public : de préférence MIT,
   Apache-2.0 ou BSD. Ultralytics est sous AGPL-3.0, donc à éviter ou à justifier. Le confier
   à `/chercheur`. Un modèle qui prend 3 images consécutives (à la TrackNet) aide pour un ballon
   petit et rapide.
3. Entraîner, avec une graine fixe, en mettant de côté une partie des images qui ne sert pas à
   l'entraînement. Mesurer sur ces images si le détecteur retrouve la position de l'ATD (rappel,
   erreur médiane en pixels) : uniquement des agrégats.
4. Lancer le détecteur seulement sur les images où l'ATD n'a pas le ballon, et garder les
   détections confiantes cohérentes avec la trajectoire (pas de téléportation du ballon).
5. Fusion dans `possession/atd.py` derrière une option, puis `possession.diagnose` (ballon vu
   en jeu), ajustement du simulateur, et `/evaluateur`.
6. Reproductibilité : le dépôt contient le code et la commande d'entraînement, et les poids sont
   publiés, par exemple dans une release GitHub, pour que les organisateurs puissent les
   télécharger. Tout est documenté dans `docs/TECHNICAL.md`.

**Règles spéciales** : l'agent écrit le code et lance les commandes, mais **n'ouvre jamais les
images, les imagettes ni les sorties image par image**. Seuls les journaux agrégés (pertes,
rappel, erreur médiane) entrent dans la conversation. L'humain vérifie visuellement dans le
visualiseur. Aucune imagette n'est corrigée ou triée à la main.

**Critère de réussite** : le « ballon vu » des images en jeu passe de ~87 % à plus de 95 %,
avec une erreur médiane de moins de 1 m sur les images mises de côté, puis le score monte au banc.

## D. Réalisme du banc d'essai (gain attendu : moyen, il conditionne tout le reste)

**Constat** : le banc donne 0,283 à la baseline, contre 0,2433 sur le vrai match. Il est donc
un peu trop facile. De plus, il simule ~65 touches par minute, contre ~43 dans l'ATD.
**Fichiers autorisés** : `validation/external.py`.
**Méthode** : calibrer uniquement sur des statistiques **agrégées** de l'ATD
(`possession.diagnose`) ou sur d'autres matchs. Le banc est bien calibré quand la baseline y
obtient environ 0,24 et que ses statistiques ressemblent à celles du diagnostic.

---

## Routine de fusion (toi, l'humain)

1. Fiche en statut READY → lire le tableau avant/après de `/evaluateur`.
2. `/auditeur-regles` doit répondre « OK pour commit ».
3. Fusionner une branche à la fois. Si deux agents ont touché le même fichier, fusionner l'une
   puis rebaser l'autre.
4. Réentraîner (`python -m validation.train --team-variant touch --learner gbdt`, sans
   `--quick`) pour régénérer `possession/weights*.json`, puis `/redacteur-doc`.
5. Vendredi 14 : gel. Samedi 15 : `/verificateur-livraison`. Dimanche 16 : soumission.

---

## E. Deuxième caméra : récupérer le ballon depuis le flux télé (gain attendu : élevé, coût moyen)

**Constat** : la caméra télé a son propre tracking. Il voit le ballon sur 50,5 % de ses images,
dans le même repère terrain (0–1) que le tracking tactique. Quand la caméra tactique perd le
ballon dans un duel, la caméra télé le voit souvent : plan plus serré, autre angle. Le règlement
autorise son usage, mais il faut retrouver soi-même l'alignement temporel entre les deux vidéos
(« mapping it back is your problem », DATA.md).

**Fichiers autorisés** : un nouveau dossier `broadcast/` (alignement + fusion),
`possession/atd.py` (fusion du ballon).

**Plan** :
1. Alignement **algorithmique** : pour chaque segment continu du flux télé (entre deux coupures),
   chercher le décalage temporel qui superpose le mieux les positions des joueurs des deux
   trackings, dans le repère terrain. Rejeter les ralentis (pas de correspondance) et les
   images dont l'homographie est invalide.
2. Mesurer la complémentarité, sur des agrégats uniquement : quelle part des images tactiques
   sans ballon a un ballon côté télé une fois l'alignement fait ?
3. Fusion : là où le ballon tactique manque et où le ballon télé existe, utiliser le ballon
   télé, en marquant sa provenance.
4. Ajuster le simulateur (`MISSING_ALIVE`) au nouveau taux de « ballon vu », réentraîner, mesurer.

**Règles** : aucune correspondance faite à la main, et l'agent ne regarde jamais les données.
Il travaille uniquement avec des statistiques agrégées (taux d'appariement, distance médiane
entre joueurs après alignement).

**Critère de réussite** : le « ballon vu » des images en jeu passe de ~87 % à plus de 93 %, puis
le score monte au banc une fois le simulateur ajusté.

---

## F. Modèle temporel profond (gain attendu : élevé ; GPU RTX 4070)

**Constat** : aujourd'hui, le modèle d'émission juge chaque image avec des fenêtres fixes (0,2 à
8 s), puis le Viterbi assemble les images avec trois coûts constants. Il ne voit donc jamais le
déroulé complet d'une action : passe, duel, récupération, 2e touche. Résultat : seulement ~44 %
des changements d'équipe tombent à ±0,5 s, et beaucoup de possessions courtes sont manquées.

**Idée** : un réseau temporel qui lit la séquence d'indices déjà calculés (`features.compute`
+ `model.team_features` + `model.dead_features`) sur ±10 s, et qui sort, pour chaque image,
P(H), P(A), P(D) ainsi que P(changement de possession ici). Ces sorties remplacent les
émissions actuelles, et le Viterbi est gardé pour la cohérence globale.
Architecture conseillée : réseau convolutif temporel dilaté (TCN) bidirectionnel, ou BiGRU,
petit (< 1 M de paramètres) : l'entraînement doit tenir en moins de 1 h sur la 4070.

**Fichiers autorisés** : `possession/seqmodel.py` (nouveau), `validation/train_seq.py`
(nouveau), `possession/emissions.py` (branchement derrière une option `kind: "seq"` dans
`weights.json`).

**Contraintes non négociables** :
* L'**inférence en numpy pur**, sans torch : les poids sont exportés en `.npz` ou `.json`, et
  conv1d, GRU et softmax sont réécrits en numpy, avec un test d'égalité numérique contre torch
  (écart maximal < 1e-5) puis un hash identique sur deux exécutions.
* Données d'entraînement : les mêmes 10 matchs du banc, dégradés (`validation/external.py`), avec
  plusieurs graines de dégradation. **Jamais le match fourni.**
* Évaluation identique au banc : un match exclu à la fois, 2 graines ; rapporter spellF1 (10
  matchs, Metrica seul, DFL seul), possessions courtes retrouvées, fausses possessions,
  changements à ±0,5 s et à ±1 s.
* Graines fixées partout. L'agent ne lit jamais les données du match.

**Critère de réussite** : spellF1 du banc ≥ 0,72 (référence v7 : 0,683) et changements
retrouvés à ±0,5 s ≥ 55 % (référence : 44 %). Échéance : vendredi midi, sinon on soumet la
meilleure version déjà validée.
