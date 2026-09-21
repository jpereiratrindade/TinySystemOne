# EXP-004: Evidência Contraditória e Robustez de Calibração (Temperature Scaling)

## Pergunta Experimental

> **Como o sistema se comporta diante de contradições factuais diretas e como a calibração pós-treino (Temperature Scaling) aprimora o Diagrama de Confiabilidade sob evidências ruidosas?**
>
> 1. Um calibrador de temperatura analítico $T > 0$ consegue reduzir o ECE para patamares ainda menores preservando $100\%$ da acurácia Top-1?
> 2. O modelo consegue discriminar entre conflitos sutis (ex: envelhecimento avançado vs saúde saudável) e conflitos diretos e severos (ex: registrado sem declaração, running com witness inválido)?

---

## Calibração Post-Hoc por Temperature Scaling

O escalonamento por temperatura aplica uma transformação estritamente monótona nos logits brutos $z$:

$$\hat{p}_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

O parâmetro de temperatura ideal $T^*$ é encontrado no conjunto de validação minimizando a Negative Log-Likelihood (NLL) via descida de gradiente exata:

$$\frac{\partial \mathcal{L}_{NLL}}{\partial T} = \frac{1}{T^2} \sum_i (\hat{p}_i(T) - y_i) z_i$$

---

## Tipos de Conflitos Avaliados

1. **Conflito de Registro/Declaração**: `declared: false` mas `registered: true`.
2. **Conflito de Testemunho/Runtime**: `runtime: running` mas `witness: invalid` (ou `signature_revoked`).
3. **Conflito de Integridade Temporal**: `health: healthy` mas `freshness: expired`.
4. **Discordância Multissensor / Assíncrona**: Sensores emitindo sinais mutuamente exclusivos.
