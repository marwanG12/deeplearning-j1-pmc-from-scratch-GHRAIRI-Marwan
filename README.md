# Deep Learning J1 — PMC from scratch

Premier reseau de neurones (Perceptron Multi-Couches) code entierement en numpy,
sans Keras pour les phases 1 a 4, puis comparaison Keras.

## Phases realisees
- **Phase 1** — Neurone unique : forward pass + loss BCE (poids fixes)
- **Phase 2** — Descente de gradient a la main, courbe de loss qui converge
- **Phase 3** — XOR resolu avec un reseau 2-2-1 + frontiere de decision
  - Note : le 2-2-1 est sensible a l'initialisation (peut rester bloque a 50%).
    J'ai ajoute des redemarrages (random restarts) pour garantir la convergence a 100%.
- **Phase 4** — Spirale 2D : reseau 2-64-64-1 avec ReLU et initialisation He,
  frontiere non-lineaire, 100% d'accuracy

## Stack
Python, numpy, matplotlib. Execute sur Google Colab.

## Auteur
Marwan Ghrairi
