# frontend — interface web d'inspection (Angular 21)

Couche frontend minimaliste au-dessus de l'API FastAPI du projet : **image → Inspect → décision**.
Aucune logique ML ici : score, seuil, décision, latence et métadonnées du modèle sont affichés tels
que renvoyés par `GET /model-info` et `POST /predict`. Documentation complète :
[`../docs/07_phase7_web_interface.md`](../docs/07_phase7_web_interface.md).

```bash
npm install            # Node 22 (si npm 10.9 échoue avec "edgesOut" : npx npm@11 install)
npm start              # http://localhost:4200 — l'API doit tourner sur http://127.0.0.1:8000
npm test               # 26 tests unitaires (vitest + jsdom), sans API
npm run build          # production → dist/frontend/browser/
npm run e2e            # bout en bout Playwright + Chrome installé (API démarrée, images de validation)
npm run format:check   # Prettier
```

- URL de l'API et limites côté client : `src/environments/environment*.ts`.
- Origines autorisées côté API (CORS) : `api.cors_origins` dans `../configs/default.yaml`.
- Structure : `src/app/core/services/inspection.service.ts` (HTTP), `src/app/core/api-error.ts`
  (messages d'erreur), `src/app/models/` (interfaces miroir de l'API), `src/app/pages/inspection/` (la page),
  `e2e/` (Playwright).
