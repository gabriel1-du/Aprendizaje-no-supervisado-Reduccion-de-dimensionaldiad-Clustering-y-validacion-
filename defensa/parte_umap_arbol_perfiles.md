# Mi parte: UMAP, árbol de decisión y perfilación (bloques 4, 5, 6)

Nota de reparto: como te tocan 4, 5 y 6, UMAP sale de la parte de la otra persona. Ella queda con 1, 2 y 3 (datos, PCA, t-SNE, agrupamiento y validación).

---

## 1. Entender el código

### Bloque 4: UMAP + K-Means

```python
df_umap = df.drop(columns="Respuesta")
reductor_umap = umap.UMAP(n_neighbors=40, min_dist=0.0, n_components=2, random_state=42)
umap_2d = reductor_umap.fit_transform(df_umap)

clusters_umap = KMeans(n_clusters=k_optimo, random_state=42, n_init=10).fit_predict(umap_2d)
```

- `drop(columns="Respuesta")`: la pauta pide excluirla, para que el espacio no dependa del resultado de la campaña.
- `n_neighbors=40`: vecinos que UMAP mira para armar el grafo. Valor alto = más estructura global; bajo = más local (el default es 15).
- `min_dist=0.0`: distancia mínima entre puntos en el mapa. Con 0, los puntos se agrupan más compactos (lo pide la pauta).
- `n_components=2`: dimensiones de salida, para graficar.
- `random_state=42`: reproducibilidad.
- K-Means sobre `umap_2d` con `k_optimo=4`: el mismo K del bloque 3.

Resultado: 589 / 783 / 542 / 291 clientes. Distinto del K-Means sobre las 31 variables (572 / 351 / 432 / 850). **Conclusión que debes decir:** el agrupamiento depende del espacio en que se hace; el principal es K-Means sobre las 31 variables y UMAP queda como comparación.

### Bloque 5: árbol de decisión

```python
arbol = DecisionTreeClassifier(random_state=42).fit(X_arbol, y_arbol)   # X = df, y = clusters_kmeans
importancias = pd.Series(arbol.feature_importances_, index=X_arbol.columns).sort_values(ascending=False)
arbol_corto = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_arbol, y_arbol)
```

- El árbol aprende a predecir la etiqueta de K-Means a partir de las variables. Se usa para **explicar** qué separa los clústeres, no para predecir.
- Árbol completo: profundidad 9, 55 hojas, exactitud 1.0. Es sobreajuste; sus importancias sirven, sus reglas no.
- Importancias: `Suma_Montos_Compras` 0.597, `Solo` 0.326, `Nro_Total_Compras` 0.013. Juntas, 93.6%.
- Árbol de profundidad 3: exactitud 0.962. Reglas (de `export_text`):
  - Sin vivir solo (`Solo` <= 0.5): si `Suma_Montos_Compras` <= 0.23 → clúster 3; si > 0.23 → clúster 0.
  - Vive solo (`Solo` > 0.5): si `Suma_Montos_Compras` <= 0.19 y `Nro_Total_Compras` <= 0.45 → clúster 2; si `Suma_Montos_Compras` <= 0.19 y `Nro_Total_Compras` > 0.45 → clúster 1; si `Suma_Montos_Compras` > 0.19 → clúster 1.
  - En la rama sin vivir solo con `Suma_Montos_Compras` <= 0.23, `Nro_Total_Compras` no cambia la clase (3 en ambos lados). Es una división redundante; decirlo si preguntan.

### Bloque 6: perfilación

```python
top3_arbol = importancias.head(3).index.tolist()
df_perfil = df.copy()
df_perfil["clusters_kmeans"] = clusters_kmeans
perfil = df_perfil.groupby("clusters_kmeans")[top3_arbol].mean()
perfil.insert(0, "n_clientes", df_perfil.groupby("clusters_kmeans").size())
```

- Agrupa por clúster y saca el promedio de las 3 variables top del árbol.
- Valores escalados entre 0 y 1: solo sirven para comparar clústeres, no tienen unidad real.
- El heatmap condensa esa tabla para leerla rápido.

| Clúster | n | Suma montos | Solo | Nro total compras | Lectura |
|---|---|---|---|---|---|
| 0 | 572 | 0.482 | 0 | 0.617 | Acompañado, gasto y compras medio-altos |
| 1 | 351 | 0.474 | 1 | 0.601 | Vive solo, gasto y compras medio-altos |
| 2 | 432 | 0.061 | 1 | 0.226 | Vive solo, gasto y compras bajos |
| 3 | 850 | 0.069 | 0 | 0.241 | Acompañado, gasto y compras bajos (el más grande) |

---

## 2. Qué decir (guion de ~5 min)

1. **UMAP:** "Excluimos `Respuesta` como pide la pauta. Con `n_neighbors=40` y `min_dist=0`, UMAP da una proyección 2D y K-Means con K=4 produce 589 / 783 / 542 / 291. Es distinto del principal, así que lo tomamos como comparación."
2. **Árbol:** "Entrenamos un árbol para explicar qué separa los clústeres. El completo sobreajusta (exactitud 1.0), así que usamos profundidad 3, con exactitud 0.962. Las variables que más pesan son el gasto acumulado (0.597) y vivir solo (0.326)."
3. **Perfiles:** "Cuatro perfiles: dos de gasto medio-alto (C0 acompañado, C1 solo) y dos de gasto bajo (C2 solo, C3 acompañado, el más grande)."
4. **Limitación (decirla tú antes de que te la pregunten):** "Las dos variables top son agregados de gasto, así que los perfiles se separan casi solo por gasto."

---

## 3. Preguntas que te pueden hacer

1. **¿Qué es `n_neighbors` y por qué 40?** Vecinos que UMAP usa para armar la estructura. Con 40 capta más estructura global que el default 15. Es el valor que pide la pauta.
2. **¿Qué es `min_dist`?** Distancia mínima entre puntos en el mapa. Con 0 los clústeres se ven más compactos; no cambia la clasificación de K-Means, solo cómo se ve el mapa.
3. **¿Por qué K-Means sobre UMAP y no UMAP directo?** UMAP reduce dimensiones pero no agrupa. El agrupamiento lo hace K-Means sobre las coordenadas. La pauta lo pide así.
4. **¿Por qué excluir `Respuesta` en UMAP?** La pauta lo pide. Además, si entra, el espacio se sesga por la aceptación de la última campaña.
5. **¿Por qué el árbol y no regresión logística?** El árbol da reglas legibles para negocio. Es el paso que pide la pauta.
6. **¿Qué es feature importance?** Cuánto reduce la impureza cada variable al dividir; se suma sobre todas las divisiones.
7. **¿Por qué el árbol completo tiene exactitud 1.0?** Memoriza los datos de entrenamiento: sobreajuste. Por eso se limita a profundidad 3 para interpretar.
8. **¿Es circular usar el árbol para explicar los clústeres?** En parte: el árbol aprende las etiquetas de K-Means. Explica la separación; no valida los clústeres. Hay que decirlo.
9. **¿Por qué `Suma_Montos_Compras` domina?** Es la suma de todos los montos, así que correlaciona con casi todos los demás. Es un agregado.
10. **¿Qué significa 0.48 en el perfil?** Valor escalado entre 0 y 1: está en la mitad del rango de gasto del dataset, no en una unidad real.
11. **¿Qué harías con más tiempo?** Quitar los agregados y repetir el árbol; probar UMAP con otros `n_neighbors`; validar perfiles con datos fuera de muestra.

---

## 4. Lo que debo poder hacer sin mirar

- Escribir de memoria la llamada de UMAP con sus 4 parámetros.
- Explicar qué devuelve `feature_importances_` y cómo se suma.
- Leer una regla del `export_text` y decir a qué clúster lleva.
- Escribir la línea de `groupby` del perfil y explicar por qué se usan promedios.
- Decir los 4 perfiles con sus cifras.

## 5. Antes de la defensa

- Ejecutar los bloques 4, 5 y 6 y confirmar que dan las mismas cifras de este archivo.
- Pedirle a la otra persona que me pregunte los bloques 1 a 3, para cubrirnos mutuamente.
