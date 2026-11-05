# Guía de defensa - EP5 (disertación de 2 personas)

Orden de trabajo: (1) entender el notebook de arriba abajo, (2) decidir qué decir en cada punto, (3) afinar conceptos y preguntas tipo.

---

## 1. Recorrido del notebook (entender antes de decir)

Ejecutar el notebook completo y recorrerlo en este orden. Para cada celda, responder en voz baja: **qué hace, qué sale y qué significa**.

| Bloque | Qué entender | Pregunta de control |
|---|---|---|
| 1. Carga | `RUTAS` busca el CSV; 2205 filas, 31 columnas, sin nulos | ¿Por qué no hay limpieza ni escalado? (la pauta lo dice: viene escalado, valores entre 0 y 1) |
| 2.1 PCA | `PCA(n_components=8)` sobre las 31 variables; varianza individual y acumulada; loadings; top 3; heatmap | ¿Qué es un loading? ¿Por qué el signo no indica importancia? |
| 2.2 t-SNE | 2 componentes, sin interpretar distancias | ¿Por qué no se puede medir la distancia entre grupos en t-SNE? |
| 3.1 Codo | SSE de K-Means para K=1..10 | ¿Qué mide la inercia? ¿Por qué el codo no es único? |
| 3.2 Silhouette | Coeficiente de -1 a 1 para K=2..10 | ¿Qué significa 0.21? ¿Por qué K empieza en 2? |
| 3.3 Selección de K | Regla de la pauta: el mayor entre codo (4) y Silhouette (2) = **4** | ¿Por qué se usa el mayor y no el más alto de Silhouette? |
| 3.4 K-Means K=4 | Tamaños 572 / 351 / 432 / 850; inercia 2467; silueta 0.215 | ¿Qué es un centroide? ¿Por qué `n_init=10`? |
| 3.5 Ward | Enlace de Ward; corte a 18.51; **4 clústeres** (respuesta a la pauta) | ¿Qué minimiza Ward? ¿Qué es la altura de corte? |
| 3.6 DBSCAN | eps=0.2, min_samples=9; 2 clústeres y **98% de ruido**; gráfico de distancia k | ¿Qué es core point, border point y ruido? ¿Por qué eps=0.2 es demasiado pequeño? |
| 3.7 Validación | Silhouette, Davies-Bouldin, Calinski-Harabasz para K-Means, Ward y DBSCAN sin ruido | ¿Por qué DBSCAN no es comparable? |
| 4. UMAP | `n_neighbors=40`, `min_dist=0`, sin `Respuesta`; K-Means sobre UMAP: 589 / 783 / 542 / 291 | ¿Qué hace `n_neighbors`? ¿Por qué se excluye `Respuesta`? |
| 5. Árbol | Árbol completo (profundidad 9, exactitud 1.0 = sobreajuste); importancias; árbol de profundidad 3 (exactitud 0.962) | ¿Por qué el árbol se usa para explicar y no para predecir? |
| 6. Perfiles | Promedios de las 3 variables top por clúster y heatmap | ¿Qué significa un valor escalado de 0.48? |

**Números que deben salir de memoria:** 80.0% (8 PC), K=4, silueta 0.215, codo sugiere 4 y Silhouette sugiere 2, Ward corta a 18.51, DBSCAN 98.0% de ruido, Davies-Bouldin 1.88 (K-Means).

---

## 2. Qué decir en cada punto (guion)

Tiempo sugerido: ~5 min por persona.

### Persona A: datos, PCA, t-SNE (bloques 1, 2, y UMAP si se asigna)
1. **Datos:** 2205 clientes, 31 variables, sin nulos, ya escaladas. No se limpia ni escala.
2. **PCA:** 8 componentes explican 80%. PC1 (25.8%) es estado civil y hogar; PC2 (19.7%) es volumen de gasto; PC3 es aceptación de campañas; PC4 es antigüedad.
3. **Limitación:** `Solo` y `ConPareja` son complementarias (r = 1.0), por eso dominan PC1 y PC2.
4. **t-SNE:** visualización 2D, no para medir distancias.

### Persona B: agrupamiento y perfiles (bloques 3, 5, 6)
1. **Elegir K:** codo sugiere 4, Silhouette sugiere 2. Regla de la pauta: el mayor, así que K=4. Admitir que la curva del codo no tiene un quiebre claro.
2. **Resultados K-Means:** 4 clústeres, silueta 0.215, validación débil pero aceptable.
3. **Ward:** corte a 18.51 da 4 clústeres (respuesta explícita a la pauta).
4. **DBSCAN:** eps=0.2 deja 98% como ruido. No coincide con codo ni Silhouette y la coincidencia no tiene valor.
5. **Árbol y perfiles:** variables top = `Suma_Montos_Compras`, `Solo`, `Nro_Total_Compras`. Los 4 perfiles:
   - C0 (572): acompañado, gasto y compras medio-altos.
   - C1 (351): vive solo, gasto y compras medio-altos.
   - C2 (432): vive solo, gasto y compras bajos.
   - C3 (850): acompañado, gasto y compras bajos (el más grande).
6. **Limitación:** las dos variables top son agregados, así que los perfiles se separan casi solo por gasto.

### Cierre (las dos personas)
- Conclusión: K-Means con K=4 es la segmentación defendible, como guía y no como grupos naturales definitivos.

---

## 3. Conceptos que deben dominar

- **PCA:** combinación lineal de variables que maximiza varianza; autovalor = varianza del componente; loading = peso de la variable en el componente; signo = dirección.
- **t-SNE:** no lineal, conserva vecindad local, no distancias globales ni densidades.
- **UMAP:** no lineal; `n_neighbors` controla vecindad local vs global; `min_dist` controla compacidad.
- **K-Means:** minimiza SSE; asume clústeres esféricos; sensible a la inicialización (`n_init`).
- **Codo:** busca el punto donde la mejora de SSE se aplana; subjetivo.
- **Silhouette:** cohesión intra-clúster vs separación inter-clúster, rango -1 a 1; 0.2 indica estructura débil.
- **Ward:** une los dos clústeres que menos aumentan la varianza intra-clúster; corte en el dendrograma define K.
- **DBSCAN:** clústeres por densidad conectada; core, border y ruido; `eps` = radio; `min_samples` = densidad mínima; no fija K.
- **Davies-Bouldin:** menor es mejor; **Calinski-Harabasz:** mayor es mejor.
- **Árbol de decisión:** divide por impureza; feature importance = reducción de impureza; sobreajuste si es profundo.

---

## 4. Preguntas tipo del profesor (con respuesta corta)

1. **¿Por qué K=4?** Codo sugiere 4, Silhouette máximo sugiere 2; la pauta dice tomar el mayor. El codo no es claro, y K=3 tiene silueta casi igual (0.215 vs 0.214).
2. **¿Por qué DBSCAN deja 98% como ruido?** eps=0.2 está muy por debajo de la mediana de distancias (0.66) y del percentil 90 (1.23). Con esos parámetros el método no segmenta.
3. **¿Qué es una silueta de 0.21?** Estructura débil: los clústeres se solapan. Por eso se presenta como guía.
4. **¿Por qué excluir Respuesta en UMAP?** La pauta lo pide; así el espacio no se sesga por el resultado de la campaña.
5. **¿Qué diferencia hay entre PCA y t-SNE?** PCA es lineal y conserva varianza global; t-SNE es no lineal y conserva vecindad local.
6. **¿Por qué Ward?** Minimiza la varianza intra-clúster al unir grupos; la altura de corte define el número de clústeres.
7. **¿Por qué `Solo` aparece tanto?** Es el complemento de `ConPareja` (r = 1.0), así que pesa en PC1 y PC2 con signo opuesto.
8. **¿Es circular el perfil?** Las variables top son agregados de gasto; los perfiles separan sobre todo por gasto. Hay que decirlo.
9. **¿Por qué el árbol completo tiene exactitud 1.0?** Sobreajuste; por eso se usa profundidad 3 (0.962) para explicar.
10. **¿Qué harían para mejorar?** Selección de características (IE2) y probar otros índices o eps en DBSCAN.

---

## 5. Puntos a afinar antes de la defensa

- Confirmar qué 4 puntos le tocan a cada uno (ver sección de asignación).
- Revisar visualmente las figuras (no las pudimos verificar en imagen).
- Completar nombres y declaración de IA en la primera celda.
- Ensayar la defensa cronometrada, cada uno con la parte del otro.
- **IE2 (15%):** la pauta pide selección de características explícita. Hoy no está en el notebook; si el profe pregunta, admitir que se decidió no hacerla y explicar qué criterio se habría usado (redundancia y varianza baja).
