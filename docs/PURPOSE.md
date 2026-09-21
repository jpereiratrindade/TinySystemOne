# TinySystemOne — Purpose & Manifesto

## Missão

**TinySystemOne** investiga se um modelo pequeno, inteiramente observável e treinável localmente pode aprender **julgamentos probabilísticos calibrados** sobre estados com evidência completa, incompleta e contraditória.

O objetivo **não** é:
- Reproduzir o Jev como cliente de API ou serviço remoto.
- Construir um LLM generativo de texto.
- Produzir métricas infladas por repetição de estados ou vazamento de dados.
- Depender de frameworks de caixa-preta.

O objetivo **é**:
- Observar como julgamento, confiança, atribuição diagnóstica e incerteza emergem do aprendizado.
- Determinar sob quais condições essas probabilidades são verdadeiramente informativas.
- Isolar a emergência do julgamento estruturado antes de introduzir complexidades de linguagem natural.

---

## Princípios Epistemológicos Centrais

1. **Julgamento Semântico ≠ Incerteza Probabilística**:
   - `Choice::Unknown` é um julgamento sobre o estado do mundo quando faltam dados. O modelo pode estar com 99.9% de certeza de que o estado é `UNKNOWN`.
   - A **Incerteza** é expressa pela **Entropia de Shannon $H(P)$** (dispersão probabilística) e pela **Incerteza Heteroscedástica $\hat{\sigma}^2$** (variância epistêmica contínua).
2. **Particionamento Estrito por Configuração Única ($\Omega$, 576 Estados)**:
   - A unidade de particionamento é o **estado estrutural canônico**, e não instâncias sorteadas com repetição.
   - É garantido matematicamente que $\text{Train} \cap \text{Test} = \emptyset$, testando generalização genuína.
3. **Locus Diagnóstico em vez de Causalidade Ilusória**:
   - `Noul` identifica a atribuição operacional da evidência (locus de anomalia) dentro da topologia do estado.
4. **Taxonomia Formal de OOD (Out-Of-Distribution)**:
   - `NO_SIGNAL`: Ausência total de sinal $\rightarrow$ requer alta entropia $H(P)$.
   - `CONTRADICTORY_EVIDENCE`: Conflito factual direto $\rightarrow$ requer decisão $P(\text{INCONSISTENT}) \rightarrow 1.0$.
   - `NOVEL_COMBINATION`: Famílias estruturais retidas do treino $\rightarrow$ teste de generalização indutiva.
   - `CORRUPTED_INPUT`: Ruído contínuo $\rightarrow$ observação dos limites de extrapolação linear.
5. **Transparência Analítica & Verificação Estrita**:
   - Matemática white-box com gradient checks analíticos vs numéricos e testes de aceitação com códigos de saída rigorosos.
