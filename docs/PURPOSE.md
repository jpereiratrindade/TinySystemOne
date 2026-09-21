# TinySystemOne — Purpose & Manifesto

## Missão

**TinySystemOne** investiga se um modelo pequeno, inteiramente observável e treinável localmente pode aprender **julgamentos probabilísticos calibrados** sobre estados com evidência completa, incompleta e contraditória.

O objetivo **não** é:
- Reproduzir o Jev.
- Construir um LLM ou modelo generativo de texto.
- Competir por benchmarks ou acurácia cega.
- Depender de frameworks pesados de ML ou serviços de rede.

O objetivo **é**:
- Observar como julgamento, confiança e incerteza emergem do aprendizado.
- Determinar sob quais condições essas probabilidades são verdadeiramente informativas.
- Isolar o aprendizado de julgamento antes de introduzir complexidades de linguagem natural.

---

## Princípios Fundamentais

1. **White-box First**: Toda a matemática (álgebra linear, ativadores, gradientes analíticos, otimizadores e métricas de calibração) é transparente, inspecionável e implementada em C++26 puro sem caixas-pretas.
2. **Independência Conceitual**: TinySystemOne não conhece serviços externos nem protocolos de terceiros.
3. **Incerteza é Cidadã de Primeira Classe**: A saída de um julgamento não é um rótulo rígido, mas uma distribuição de probabilidade com métricas explícitas de confiança, entropia e erro de calibração (ECE / Brier Score).
4. **Isolamento de Fenômenos**: Primeiro prova-se o fenômeno de julgamento e calibração em estados estruturados (v0.1); apenas em versões futuras avança-se para embeddings, atenção e linguagem semi-estruturada.
