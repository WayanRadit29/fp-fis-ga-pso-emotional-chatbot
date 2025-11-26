# fp-fis-ga-pso-emotional-chatbot

# FP: Optimasi FIS untuk Chatbot Emosional (GA vs PSO)

**Tujuan**: Membandingkan Genetic Algorithm (GA) dan Particle Swarm Optimization (PSO) untuk mengoptimasi parameter Membership Function pada Fuzzy Inference System (FIS) yang menentukan *tone* respon chatbot.

## Dataset
- Hugging Face: `dair-ai/emotion` (label emosi → dimapping ke tone chatbot).

## Arsitektur Singkat
1) NLP → skor probabilitas emosi  
2) FIS (Mamdani, trimf; 4 input, 1 output)  
3) Optimasi parameter MF: GA vs PSO  
4) Evaluasi: F1-weighted, akurasi, waktu, stabilitas
