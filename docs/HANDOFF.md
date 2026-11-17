# Passation — état au lundi 16 novembre, 22 h 30 (heure de Québec)

**Date limite : mardi 17 novembre, 23 h 59 CET = 17 h 59 à Québec.** Le lien du formulaire de
soumission n'a pas encore été reçu (il arrive à l'adresse d'inscription, pas sur Gmail) : il a été
demandé aux organisateurs. Le formulaire est modifiable ; la dernière réponse compte.

## 1. Règles (courriel de clarification des organisateurs, prioritaire sur tout le reste)

Interdit :
- vérifier nos sorties contre la vidéo (donc **ne plus utiliser `tools/viewer.html`**) ;
- étiquettes ou données humaines sur le match fourni ;
- envoyer les données fournies à un service d'IA (**aucun agent ne lit `data/` ni `cache/`**) ;
- entraîner, ajuster ou calibrer sur la vérité terrain ou sur la vidéo fournie.

Obligatoire : tout entraînement et tout réglage de paramètre se font **sur des données externes**.
Permis : vérifications automatiques sur les données fournies ; tout jeu de données dont on a les
droits (vérifier la licence).

Conséquences déjà tirées — **ne pas relancer** :
- Fiche C (détecteur de ballon entraîné sur la vidéo fournie) : interdite, et avait échoué (1,2 %).
- Fiche H (auto-apprentissage sur le match fourni) : interdite.
- Fiche E (ballon de la caméra télé) : permise mais gain mesuré trop faible (+1,5 point de ballon vu).
- Les 5 minutes de vérité publique : vérification seulement, jamais pour choisir une version.

## 2. Version prête à soumettre : branche `clean-trees`

- Modèle : arbres boostés (variante `duel_past`, pondération par possession) + Viterbi.
- Entraînement : Metrica (3 matchs) + IDSSE Bundesliga (7 matchs, CC BY 4.0). Pas de PFF.
- Simulateur `validation/external.py` : constantes tirées de données externes uniquement ; arrêts
  typés (ballon sorti / ballon resté sur le terrain).
- Fichier : `submissions/TercerHombre.csv`, SHA-256 `BEF80E4797BAF6803FF2506E803320D4015DB98FED2381A27B898FD092D05F9B`
  (régénéré deux fois à l'identique). Document : `TercerHombre.md`.
- Dépôt à déclarer : `https://github.com/TEKAB2/more-than-a-hack-2027/tree/clean-trees`.

## 3. Chiffres — ne jamais confondre les deux colonnes

| version | banc d'essai (10 autres matchs, matchs exclus) | vrai match, 5 min publiques (17 possessions, très bruité) |
|---|---|---|
| baseline officielle | 0,26–0,28 | 0,243 sur le match complet |
| v7 (arbres, ancien simulateur) | 0,683 | 0,565 |
| ensemble 5 réseaux + PFF (`main`) | 0,775 **non fiable** (réseaux entraînés sur les matchs du banc) | 0,575 |
| **clean-trees** | 0,674 | **0,594** |

Le banc surestime le vrai match d'environ 0,10. Défaut principal sur le vrai match : seulement
34 % des images de ballon mort reconnues (le banc en annonce 94 % : il ne reproduit pas les vrais arrêts).
Borne mesurée au banc : équipe parfaite → 0,953 ; ballon mort parfait → 0,711.

## 4. Pistes en cours (à mesurer au banc, en matchs exclus, avant toute adoption)

1. **SkillCorner Open Data** (10 matchs, tracking tiré de la vidéo télé, possession par image) :
   vérifier la licence des données ; l'utiliser tel quel, sans dégradation ; adopter seulement si
   le score sur les 3 matchs Metrica (exclus) monte d'au moins 0,005.
2. **Un réseau 102 canaux réentraîné sur le nouveau simulateur**, chaque pli sans le match testé,
   un seul modèle en mémoire (la machine a manqué de mémoire avec tout chargé). Adopter si
   arbres + réseau gagne au moins 0,01 sur arbres seuls (0,674).
3. PFF : formulaire officiel rempli ; licence non explicite ; question posée aux organisateurs.
   Sans réponse positive, ne pas soumettre de modèle entraîné sur PFF.

## 5. Avant de remplacer la soumission

- pipeline lancé deux fois → même SHA-256 ; `evaluation/validate_submission.py` → Valid ;
- `git status` : rien sous `data/`, `cache/`, `submissions/`, `validation/external_data/` ;
- `TercerHombre.md` décrit exactement le modèle soumis et cite toutes les données utilisées ;
- inférence en numpy pur, sans réseau ni pickle ; aucune constante mesurée sur le match fourni.

## 6. Questions envoyées aux organisateurs (réponses attendues)

Lien du formulaire ; droit d'utiliser PFF ; éligibilité malgré l'usage du visualiseur avant la clarification.
