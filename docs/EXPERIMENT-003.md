# EXP-003: Missing Values Explícitos e Aprendizado de Incerteza

## Pergunta Experimental

> **Como um modelo neural pequeno reage sob omissão progressiva de evidência ($0\% \rightarrow 100\%$ missingness)?**
>
> 1. Ele aprende a usar os canais de presença ($m_k \in \{0, 1\}$) para reconhecer a ausência de dados?
> 2. A incerteza heteroscedástica $\hat{\sigma}^2$ e a entropia de Shannon $H(P)$ crescem de forma monotônica conforme campos são ocultados?
> 3. Sob $100\%$ de omissão (vazio total), o modelo converge confiavelmente para o julgamento `UNKNOWN`, locus `MULTIPLE` e variância máxima?

---

## Representação de Entrada (24 dimensões)

| Segmento | Dimensões | Descrição |
| :--- | :---: | :--- |
| `Features` | 18 | Categorias One-Hot ativas (zeradas caso o campo esteja missing) |
| `Presence Mask` | 6 | Bits booleanos de presença dos 6 atributos (`declared`, `registered`, `runtime`, `witness`, `freshness`, `health`) |

---

## Arquitetura Multi-Head com Incerteza (4 Cabeças)

- **Trunk**: $24 \rightarrow 32 \text{ (GELU)} \rightarrow 16 \text{ (GELU)}$
- **Choice Head**: $16 \rightarrow 4 \text{ (Softmax)}$
- **Noul Head**: $16 \rightarrow 7 \text{ (Softmax)}$
- **Score Head**: $16 \rightarrow 1 \text{ (Sigmoid)}$ $\rightarrow \hat{\mu}$
- **Uncertainty Head**: $16 \rightarrow 1 \text{ (Softplus)}$ $\rightarrow \hat{\sigma}^2$

### Função de Perda com Gaussian NLL

$$\mathcal{L}_{total} = \mathcal{L}_{choice} + \mathcal{L}_{noul} + \mathcal{L}_{NLL}(\hat{\mu}, \hat{\sigma}^2, y_{score})$$

onde:

$$\mathcal{L}_{NLL}(\mu, \sigma^2, y) = \frac{(\mu - y)^2}{2\sigma^2} + \frac{1}{2}\ln(\sigma^2 + \epsilon)$$

---

## Protocolo de Avaliação de Missingness

Testamos o mesmo conjunto de estados com taxas controladas de omissão artificial:
- **0% Missing**: Evidência completa.
- **20% Missing**: 1 campo omitido aleatoriamente.
- **40% Missing**: 2 campos omitidos.
- **60% Missing**: 3 campos omitidos.
- **80% Missing**: 4 campos omitidos.
- **100% Missing**: Todos os 6 campos omitidos (zero sinal).
