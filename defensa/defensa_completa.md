# Defensa completa - EP5 (todos los bloques del notebook)

Cómo usar esta guía: primero recorre cada bloque del notebook y comprueba que entiendes lo que hace el código. Después lee "Qué decir" y las preguntas de ese bloque. Al final, la sección de reparto.

Cifras clave que deben salir de memoria:
- 2205 clientes, 31 variables, sin nulos.
- PCA: 8 componentes explican **80.0%**.
- Codo sugiere **K=4**; Silhouette máximo en **K=2** (0.275); se elige **K=4** por la regla de la pauta.
- K-Means K=4: 572 / 351 / 432 / 850; silueta **0.215**.
- Ward: corte a **18.51**, **4 clústeres**: 361 / 421 / 961 / 462.
- DBSCAN (eps 0.2, min_samples 9): 2 clústeres (31 y 14 puntos), **98.0% de ruido**.
- Validación K-Means: Davies-Bouldin 1.88, Calinski-Harabasz 502.
- UMAP + K-Means K=4: 589 / 783 / 542 / 291.
- Árbol completo: profundidad 9, exactitud 1.0 (sobreajuste). Árbol de profundidad 3: exactitud 0.962.
- Perfiles: C0 572, C1 351, C2 432, C3 850.

---

## Bloque 1: Carga de datos y librerías

**Código:** importa pandas, numpy, scikit-learn, umap, seaborn. Busca el CSV en `../utils/`, `utils/` o la carpeta actual (para que funcione en el repo y en Colab). Imprime filas, columnas y nulos.

**Qué decir:** "El dataset tiene 2205 clientes y 31 variables, sin valores nulos. Viene escalado entre 0 y 1, así que no aplicamos limpieza ni escalado, como indica la pauta."

**Preguntas:**
- ¿Por qué no escalar? Ya viene escalado; escalar otra vez no cambia el orden de las distancias pero sí complica la lectura.
- ¿Qué tipo de variables hay? Casi todas numéricas continuas (montos, edad) y binarias (`Solo`, `ConPareja`, aceptación de campañas, `Respuesta`).

---

## Bloque 2.1: PCA

**Código:**
```python
pca = PCA(n_components=8, random_state=42)
pca.fit(df)
```
- Calcula 8 componentes sobre las 31 variables.
- `explained_variance_` = autovalor (varianza del componente). `explained_variance_ratio_` = proporción.
- `components_.T` = matriz de cargas (loadings): variables en filas, componentes en columnas.
- El top 3 ordena cada columna por valor absoluto.
- El heatmap muestra las cargas con colores divergentes (rojo/azul = signo).

**Resultados:**
| PC | Varianza | Acumulada |
|---|---|---|
| PC1 | 25.8% | 25.8% |
| PC2 | 19.7% | 45.6% |
| PC3 | 8.9% | 54.5% |
| PC4 | 6.7% | 61.2% |
| PC5 | 6.1% | 67.4% |
| PC6 | 4.8% | 72.1% |
| PC7 | 4.0% | 76.1% |
| PC8 | 3.9% | **80.0%** |

Top 3 por componente:
- **PC1:** `ConPareja` 0.661, `Solo` -0.661, `Tamano_Familia` 0.227. Eje de estado civil y hogar.
- **PC2:** `Suma_Montos_Compras` 0.353, `Nro_Total_Compras` 0.304, `MontoVinos` 0.302. Eje de gasto y volumen.
- **PC3:** `Respuesta` 0.675, `Nro_Campanas_Aceptadas` 0.297, `AceptacionCampana3` 0.252. Eje de aceptación de campañas.
- **PC4:** `Antiguedad` 0.861, `CantAdolescentes` 0.289. Eje de fidelidad temporal.
- **PC5:** `CantAdolescentes` 0.572, `Educacion` 0.429, `Antiguedad` -0.321. Eje de hogar y educación.
- **PC6:** `DiasUltCompra` 0.779, `AceptacionCampana4` 0.388. Eje de recencia.
- **PC7 y PC8:** mezcla de campañas, recencia, educación y adolescentes. No forzar una interpretación.

**Qué decir:** "Ocho componentes explican 80% de la información. PC1 y PC2 separan hogar y gasto; PC3 separa la respuesta a campañas; PC4 la antigüedad. Los últimos componentes mezclan variables y no tienen un nombre claro."

**Preguntas:**
- ¿Qué es un loading? El peso de cada variable en el componente. Su signo indica dirección, no importancia.
- ¿Por qué 8 componentes? Lo pide la pauta; con 8 se retiene 80%.
- ¿Por qué `ConPareja` domina PC1 y PC2? Porque `Solo` es su complemento exacto (correlación 1.0): tienen el mismo peso con signo opuesto. Es una limitación del dataset; decirlo.
- ¿Por qué PC3 tiene `Respuesta`? Porque `Respuesta` entra en el conjunto de variables de PCA (la pauta solo la excluye en UMAP).

---

## Bloque 2.2: t-SNE

**Código:** `TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")` sobre las 31 variables, y un scatter 2D.

**Qué decir:** "t-SNE es no lineal y conserva la vecindad local. Lo usamos para visualizar la distribución, no para medir distancias entre grupos."

**Preguntas:**
- ¿Por qué no sirve para medir distancias entre grupos? Porque conserva vecindad local, no distancias globales ni densidades absolutas.
- ¿Qué es `perplexity`? Cuántos vecinos efectivos considera cada punto (default 30). Cambia la forma del gráfico.
- ¿PCA o t-SNE? PCA es lineal y conserva varianza global; t-SNE es no lineal y sirve para ver estructura local.

---

## Bloque 3.1: Método del codo

**Código:** `KMeans(n_clusters=k, n_init=10)` para K=1..10 y la inercia (SSE).

Reducción del SSE al agregar un clúster: K=2 1053, K=3 397, K=4 238, K=5 178, K=6 98, K=7 97, K=8 55.

**Qué decir:** "La caída es fuerte hasta K=4 y se suaviza después. El codo no es tan marcado como para fijar K sin discusión; lo ubicamos en K=4."

**Preguntas:**
- ¿Qué mide la inercia? La suma de distancias al cuadrado de cada punto a su centroide. Baja siempre al subir K.
- ¿Por qué el codo es subjetivo? No hay una regla única para encontrar el punto de quiebre.
- ¿Por qué `n_init=10`? K-Means depende de la inicialización; se corre 10 veces y se queda con la mejor.

---

## Bloque 3.2: Coeficiente de Silhouette

**Código:** `silhouette_score` para K=2..10 con K-Means.

Valores: K=2 **0.275**, K=3 0.215, K=4 0.214, K=5 0.203, K=6 0.192, K=7 0.200, K=8 0.153.

**Qué decir:** "El máximo está en K=2. Todos los valores quedan por debajo de 0.3, así que los clústeres se solapan y la estructura es débil."

**Preguntas:**
- ¿Qué mide Silhouette? Para cada punto, `(b − a) / max(a, b)`, donde `a` es la distancia media dentro de su clúster y `b` la del clúster vecino más cercano. Va de -1 a 1.
- ¿Por qué K empieza en 2? Silhouette no se define con un solo clúster.
- ¿Qué significa 0.21? Estructura débil: el punto medio está entre clústeres.

---

## Bloque 3.3: Selección de K

**Código:** `k_codo = 4`, `k_silhouette = argmax(silhouette)` = 2, `k_optimo = max(4, 2) = 4`.

**Qué decir:** "La pauta dice que, si codo y Silhouette no coinciden, se elige el mayor. El codo sugiere 4 y Silhouette 2, así que K=4."

**Preguntas:**
- ¿Por qué el mayor y no el de mejor Silhouette? Lo dice la pauta; hay que citarla.
- ¿Y si el profe pregunta por K=3? Silhouette de K=3 (0.215) es casi igual al de K=4 (0.214). Es defendible; decirlo.

---

## Bloque 3.4: K-Means con K=4

**Código:**
```python
kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
clusters_kmeans = kmeans.fit_predict(df)
```
Gráfico PC1-PC2 con centroides proyectados.

Resultado: 572 / 351 / 432 / 850 clientes. Inercia 2467. Silhouette 0.215.

**Qué decir:** "K-Means con K=4 separa los clientes en cuatro grupos de tamaños desiguales. Proyectados en PC1-PC2, los grupos se solapan parcialmente."

**Preguntas:**
- ¿Qué es un centroide? El promedio de los puntos de su clúster.
- ¿Qué supone K-Means? Clústeres esféricos y de tamaño parecido; por eso puede fallar con grupos alargados.
- ¿Por qué el gráfico PC1-PC2? Es el plano de mayor varianza; se usa solo para visualizar.

---

## Bloque 3.5: Jerárquico (Ward)

**Código:**
```python
Z = linkage(df, method="ward")
umbral = (Z[-4, 2] + Z[-3, 2]) / 2
etiquetas_jerarquico = fcluster(Z, t=umbral, criterion="distance")
```
Dendrograma con truncado (últimas 30 hojas) y línea de corte.

Resultado: corte a **18.51**, 4 clústeres, tamaños 361 / 421 / 961 / 462.

**Qué decir (respuesta a la pauta):** "Con la altura de corte elegida se identifican **4 clústeres**. El corte se pone entre la antepenúltima y la penúltima fusión, para que queden 4 grupos."

**Preguntas:**
- ¿Qué minimiza Ward? El aumento de varianza intra-clúster al unir dos grupos.
- ¿Qué es la altura de corte? La distancia en la que se separan los grupos; cortar más arriba da menos grupos.
- ¿Por qué no coinciden los tamaños con K-Means? Son algoritmos distintos; las particiones no tienen por qué coincidir.

---

## Bloque 3.6: DBSCAN

**Código:**
```python
dbscan = DBSCAN(eps=0.2, min_samples=9)
etiquetas_dbscan = dbscan.fit_predict(df)
```
Gráfico de distancia k con `NearestNeighbors(n_neighbors=9)`: mediana 0.66, percentil 90 = 1.23, línea en eps = 0.2.

Resultado: 2 clústeres (31 y 14 puntos), **2160 de 2205 (98.0%) como ruido**.

**Qué decir:** "Con eps=0.2 el método no segmenta la base: 98% queda como ruido. eps está muy por debajo de la distancia típica entre clientes (mediana 0.66), así que el resultado no describe a la población."

**Respuesta a la pauta (¿misma cantidad de clústeres?):** "No coincide: el codo sugiere 4, Silhouette 2 y DBSCAN 2. La coincidencia con Silhouette no tiene valor, porque los métodos son distintos."

**Preguntas:**
- ¿Qué es core point, border point y ruido? Core: al menos `min_samples` vecinos dentro de `eps`. Border: vecino de un core pero sin densidad propia. Ruido: ninguno de los dos.
- ¿Qué pasa si subo eps? Más puntos entran en clústeres; la solución cambia. No lo hicimos porque la pauta fija 0.2.
- ¿Por qué el método no fija K? Define clústeres por densidad conectada, no por partición.

---

## Bloque 3.7: Validación interna (IE5 e IE6)

**Código:** `silhouette_score`, `davies_bouldin_score` y `calinski_harabasz_score` para K-Means, Ward y DBSCAN sin ruido.

| Algoritmo | Clústeres | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|---|
| K-Means | 4 | 0.215 | 1.88 | 502 |
| Jerárquico Ward | 4 | 0.178 | 2.09 | 437 |
| DBSCAN (sin ruido, 45 puntos) | 2 | 0.394 | 1.00 | 31 |

**Qué decir (juicio de calidad):** "K-Means con K=4 es el mejor de los tres en los tres índices. La calidad es aceptable pero débil. DBSCAN parece mejor, pero se calcula sobre el 2% de los datos, así que no es comparable."

**Preguntas:**
- ¿Qué mide Davies-Bouldin? Similitud promedio entre cada clúster y su más parecido. Menor es mejor.
- ¿Qué mide Calinski-Harabasz? Razón entre dispersión entre clústeres e intra-clúster. Mayor es mejor.
- ¿Por qué DBSCAN parece mejor? Porque al quitar el ruido se calcula sobre pocos puntos muy densos; el resultado es engañoso.

---

## Bloque 4: UMAP + K-Means

**Código:**
```python
df_umap = df.drop(columns="Respuesta")
reductor_umap = umap.UMAP(n_neighbors=40, min_dist=0.0, n_components=2, random_state=42)
umap_2d = reductor_umap.fit_transform(df_umap)
clusters_umap = KMeans(n_clusters=k_optimo, random_state=42, n_init=10).fit_predict(umap_2d)
```
- Excluye `Respuesta` (pide la pauta).
- `n_neighbors=40`: vecinos del grafo; más alto = más estructura global.
- `min_dist=0.0`: puntos más compactos en el mapa.
- `n_components=2`: para graficar.

Resultado: 589 / 783 / 542 / 291.

**Qué decir:** "UMAP reduce a 2D conservando estructura local y parte de la global. K-Means sobre esas coordenadas, con K=4, da tamaños distintos del principal. Por eso UMAP es comparación; el principal es K-Means sobre las 31 variables."

**Preguntas:**
- ¿Qué hace `n_neighbors`? Cuántos vecinos mira para armar la estructura. Valor alto = más global.
- ¿Qué hace `min_dist`? Distancia mínima entre puntos en el mapa. Solo cambia la visualización, no la clasificación.
- ¿UMAP agrupa? No; reduce dimensiones. El agrupamiento lo hace K-Means.
- ¿Por qué excluir `Respuesta`? La pauta lo pide, y evita que el espacio se sesgue por la campaña.

---

## Bloque 5: Árbol de decisión

**Código:**
```python
arbol = DecisionTreeClassifier(random_state=42).fit(df, clusters_kmeans)
importancias = pd.Series(arbol.feature_importances_, index=df.columns).sort_values(ascending=False)
arbol_corto = DecisionTreeClassifier(max_depth=3, random_state=42).fit(df, clusters_kmeans)
```

- Árbol completo: profundidad 9, 55 hojas, exactitud 1.0 (sobreajuste).
- Importancias: `Suma_Montos_Compras` 0.597, `Solo` 0.326, `Nro_Total_Compras` 0.013. Top 3 = 93.6%.
- Árbol de profundidad 3: exactitud 0.962. Reglas:
  - Sin vivir solo (`Solo` <= 0.5): `Suma_Montos_Compras` <= 0.23 → clúster 3; > 0.23 → clúster 0.
  - Vive solo (`Solo` > 0.5): `Suma_Montos_Compras` <= 0.19 y `Nro_Total_Compras` <= 0.45 → clúster 2; `Suma_Montos_Compras` <= 0.19 y `Nro_Total_Compras` > 0.45 → clúster 1; `Suma_Montos_Compras` > 0.19 → clúster 1.
  - Una división no cambia la clase (`Nro_Total_Compras` bajo `Suma_Montos_Compras` <= 0.23): redundante.

**Qué decir:** "Entrenamos un árbol para explicar qué separa los clústeres, no para predecir. El completo sobreajusta, así que interpretamos el de profundidad 3, con 0.962 de exactitud. Lo que más separa es el gasto acumulado y vivir solo."

**Preguntas:**
- ¿Por qué no predecir con el árbol? Porque lo que se quiere es explicar la segmentación.
- ¿Qué es feature importance? Cuánto reduce la impureza cada variable, sumado sobre todas sus divisiones.
- ¿Por qué el completo tiene exactitud 1.0? Memoriza los datos: sobreajuste.
- ¿Es circular? En parte: el árbol aprende las etiquetas de K-Means. Explica la separación; no valida el clustering.

---

## Bloque 6: Perfilación

**Código:**
```python
top3_arbol = importancias.head(3).index.tolist()
df_perfil = df.copy()
df_perfil["clusters_kmeans"] = clusters_kmeans
perfil = df_perfil.groupby("clusters_kmeans")[top3_arbol].mean()
perfil.insert(0, "n_clientes", df_perfil.groupby("clusters_kmeans").size())
```
Heatmap de esos promedios.

| Clúster | n | Suma montos | Solo | Nro total compras | Lectura |
|---|---|---|---|---|---|
| 0 | 572 | 0.482 | 0 | 0.617 | Acompañado, gasto y compras medio-altos |
| 1 | 351 | 0.474 | 1 | 0.601 | Vive solo, gasto y compras medio-altos |
| 2 | 432 | 0.061 | 1 | 0.226 | Vive solo, gasto y compras bajos |
| 3 | 850 | 0.069 | 0 | 0.241 | Acompañado, gasto y compras bajos (el más grande) |

**Qué decir:** "Cuatro perfiles. Dos tienen gasto medio-alto: C0 con pareja y C1 que vive solo. Dos tienen gasto bajo: C2 que vive solo y C3 con pareja, que es el más grande. Los valores están escalados entre 0 y 1, así que comparan clústeres, no montos reales."

**Limitación (decirla tú):** "Las dos variables de gasto son agregados, así que los perfiles se separan casi solo por gasto."

**Preguntas:**
- ¿Qué significa 0.48 en el perfil? Valor escalado entre 0 y 1: mitad del rango del dataset, no una unidad real.
- ¿Por qué promedios y no medianas? La pauta permite ambos; los promedios son más fáciles de leer.
- ¿Qué harías con más tiempo? Quitar agregados, repetir el árbol y validar fuera de muestra.

---

## Introducción y cierre (las dos personas)

**Introducción (1 min):** "Segmentamos 2205 clientes de retail con 31 variables de compra. Usamos reducción de dimensionalidad, tres algoritmos de agrupamiento y validación interna para decidir un segmento defendible."

**Cierre (1 min):** "Conclusión: K-Means con K=4 es la segmentación más defendible. La silueta es baja, así que los grupos sirven como guía para campañas y no como grupos naturales definitivos. Como mejora, quitaríamos agregados y haríamos selección de características."

**Riesgo IE2 (15%):** la pauta pide selección de características explícita. El notebook no la tiene. Si preguntan: "Se decidió no hacerla; el criterio que habríamos usado es redundancia (|r| > 0.8) y varianza baja." Admitirlo es mejor que improvisar.

---

## Reparto sugerido (para dividir con la compañera)

- **Persona A:** introducción, bloques 1, 2.1, 2.2 y 3.1 a 3.3 (datos, PCA, t-SNE, codo y Silhouette).
- **Persona B:** bloques 3.4 a 3.7, 4, 5, 6 y cierre (K-Means, Ward, DBSCAN, validación, UMAP, árbol, perfiles).

Cada una debe poder responder las preguntas de la otra parte en 2 minutos sin notas.
