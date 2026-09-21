# EXP-005: Detecção de OOD por Energia Livre, Predição Seletiva e Generalização Extrema

## Pergunta Experimental

> **Como um modelo neural white-box pode detectar com robustez que uma entrada está Fora da Distribuição (OOD) e abster-se formalmente de decisões superconfiantes em dados corrompidos ou alienígenas?**
>
> 1. A métrica de **Energia Livre de Helmholtz** $E(x; T) = -T \ln \sum \exp(z_i / T)$ separa de forma confiável distribuições legítimas de ruídos alienígenas (AUROC $\ge 0.95$)?
> 2. O mecanismo de **Predição Seletiva com Abstenção** (`SelectivePredictor`) consegue barrar a extrapolação cega, forçando alta incerteza e $H(P) = 1.0$ em casos espúrios?

---

## Formulação da Energia Livre de Helmholtz

Em modelos discriminativos parametrizados com Softmax, a densidade conjunta não normalizada $p(x, y)$ relaciona-se diretamente com a energia livre:

$$E(x; T) = -T \cdot \ln \sum_{i=1}^K \exp\left(\frac{z_i(x)}{T}\right)$$

- **Entradas In-Distribution (ID)**: Possuem logits bem estruturados com alta magnitude na classe correta $\rightarrow$ Energia baixa / negativa ($E(x) \ll 0$).
- **Entradas Out-Of-Distribution (OOD)**: Possuem ativações desbalanceadas ou fracas $\rightarrow$ Energia alta ($E(x) > \tau$).

---

## Mecanismo de Abstenção Formal

Dado o limiar calibrado no 95º percentil de validação $\tau_{OOD}$:

$$\text{Decisão}(x) = \begin{cases} 
\hat{y}(x), \quad \text{se } E(x) \le \tau_{OOD} & \text{(Aceito: Julgamento Normal)} \\ 
\text{ABSTAIN} \ (\text{UNKNOWN}, \ H_{norm}=1.0), \quad \text{se } E(x) > \tau_{OOD} & \text{(Abstenção: Fora de Domínio)} 
\end{cases}$$

---

## Suites de Benchmark OOD Avaliadas

1. **ID_Test**: 80 estados canônicos disjuntos (nunca vistos no treino).
2. **OOD_NoSignal**: Vetor nulo (zero evidência).
3. **OOD_UniformDispersion**: Distribuição uniforme plana.
4. **OOD_AlienNoise**: Ruídos gaussianos contínuos e magnitudes corrompidas.
5. **OOD_NovelCombination**: Famílias de estados deliberadamente retidas do treinamento.
