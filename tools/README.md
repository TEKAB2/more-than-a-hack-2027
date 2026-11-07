# Visualiseur de possession

Outil local pour **regarder** ce que fait le pipeline : la vidéo tactique, les joueurs et le ballon
du tracking projetés sur l'image, et les étiquettes H/A/D sous forme de timeline.

## Utilisation

```powershell
python -m tools.export_viewer          # -> cache/viewer_pack.bin (~40 Mo, quelques secondes)
```

Ouvre ensuite `tools/viewer.html` dans **Chrome ou Edge** (double-clic) et charge :

1. **Vidéo** : `data/tactical/BAR-ATH_tactical.mp4`
2. **Pack** : `cache/viewer_pack.bin`
3. **CSV** (plusieurs à la fois) : `submissions/TercerHombre.csv`, une ancienne version pour
   comparer, et `evaluation/gt/BAR-ATH_possession_gt_public.csv` (la vérité des 5 premières minutes).

La ligne 1 de la timeline est le premier CSV chargé ; « désaccord » compare les lignes 1 et 2.

## Ce qu'on voit

* **Sur la vidéo** : pastilles bleues (Barça), rouges (Athletic), jaune (ballon), placées par
  l'homographie. Le bandeau jaune « ballon non suivi » indique les frames où le tracking a perdu
  le ballon. En haut à droite : l'étiquette de la ligne 1.
* **Timeline du match** : les étiquettes de chaque CSV sur tout le match, et les coups de pied
  arrêtés du Smart Tagging. Clique pour te déplacer.
* **Timeline zoomée** (molette ou `+`/`−`) : les étiquettes image par image, une barre blanche à
  chaque changement, la présence du ballon (jaune), les coups de pied arrêtés (orange) et les
  relances détectées (triangles blancs), puis ce que « pense » le modèle : la courbe blanche est
  P(Barça | en jeu) (haut = Barça, bas = Athletic), la zone grise est P(ballon mort).

## Règles à respecter

* Tout reste sur ton ordinateur : la page n'a aucun code réseau. Le pack est dans `cache/`, ignoré
  par git. **Ne le publie jamais et ne l'envoie à personne** (ni à une IA) : il contient des
  positions image par image tirées des données sous licence.
* Regarder pour **comprendre** est permis. Transformer ce qu'on voit en étiquettes, en corrections
  ou en constantes liées à des frames précises est interdit (RULES.md). Les notes servent à décrire
  des **tendances** d'erreur (« il donne la possession à l'équipe qui dégage de la tête »), qu'on
  corrige ensuite de façon générale dans le code.
