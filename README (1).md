# Plateforme de recommandation éducative — Démo PFE

Démonstration interactive du système de recommandation éducative
*context-aware* et *scalable* (atténuation de la sparsité, du démarrage à
froid, et passage à l'échelle).

L'application présente trois volets :

1. **Recommandation (cas nominal)** — *Learning-to-Rank* (Gradient Boosting,
   Config D) pour un apprenant disposant d'un historique.
2. **Démarrage à froid** — apprenant nouveau (DKT) et cours nouveau (EERNN-M,
   transfert par contenu sémantique).
3. **Architecture & preuves** — pipeline Big Data (Medallion Bronze→Silver→Gold)
   et résultats clés.

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

L'application charge ses modèles depuis le dossier `./artefacts/`.

## Structure du dépôt

```
.
├── app.py                  # application Streamlit
├── requirements.txt
├── .streamlit/config.toml  # thème
└── artefacts/              # modèles et données de démonstration
    ├── dkt_user.pt, dkt_user_meta.json, course_mapping_user.csv
    ├── eernn_item.pt, eernn_item_meta.json, S1_embeddings.npy, course_mapping_item.csv
    ├── gb_ltr_D.pkl, gb_ltr_features.json, gb_ltr_test.csv
    └── ltr_feature_families.json, ltr_importance.json (optionnels)
```

## Déploiement

Déployé via [Streamlit Community Cloud](https://share.streamlit.io) à partir
de ce dépôt. Le fichier principal est `app.py`.

> Les versions de `scikit-learn` et `torch` dans `requirements.txt` doivent
> correspondre à celles ayant produit les artefacts, afin de garantir le
> chargement des modèles sérialisés.
