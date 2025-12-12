# Deteccao-onnx

Aplicação em **C# (Windows Forms)** para realizar **detecção de objetos usando um modelo ONNX** (exportado de YOLO) em imagens estáticas.  
O projeto foi desenvolvido como um exemplo prático de integração entre **rede neural treinada em Python** e uma interface gráfica em C# para uso em ambiente Windows. 

---

## 📁 Estrutura do Projeto

Repositório:

- `treinamento/` – Projeto principal em C# (Windows Forms)
- `treinamento.slnx` – Solução do Visual Studio
- `.gitignore` / `.gitattributes` – Arquivos de configuração do Git 

Dentro do projeto `treinamento` (padrão de app WinForms):

- `Form1.cs` – Lógica principal da interface e da inferência ONNX
- `Program.cs` – Ponto de entrada da aplicação
- Outros arquivos gerados pelo Visual Studio (Designer, recursos, etc.)

> ⚠️ O caminho do modelo ONNX é configurado diretamente no código (campo `ModelPath` em `Form1.cs`).

---

## 🧠 Funcionalidades

- Carregamento de uma imagem (`.jpg`, `.jpeg`, `.png`, `.bmp`) via `OpenFileDialog`.
- Pré-processamento da imagem para o tamanho esperado pelo modelo (por exemplo, `512x512`).
- Execução do modelo ONNX via **Microsoft.ML.OnnxRuntime**.
- Decodificação da saída no formato YOLO (cx, cy, w, h, classes…).
- Aplicação de **Non-Maximum Suppression (NMS)** para remover caixas sobrepostas.
- Desenho das bounding boxes e labels diretamente sobre a imagem:
  - Caixa colorida (por padrão, vermelho).
  - Texto com `Classe` e `Score` (probabilidade).
- Lista textual das detecções em um `ListBox` (coordenadas e confiança).
- Ajuste de limiar de confiança via `TextBox` na interface.

---

## 🛠️ Tecnologias e Bibliotecas

- **Linguagem:** C#   
- **Tipo de aplicação:** Windows Forms
- **Framework .NET:** .NET 8.0 (ou compatível, dependendo da sua configuração local)
- **Inferência de modelo:**  
  - [Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/)
- **Modelo de IA:**  
  - Arquivo ONNX (por exemplo `best_ir11.onnx`), exportado de YOLOv8/YOLO11 (Ultralytics) ou similar.

---

## ✅ Pré-requisitos

1. **Sistema operacional:** Windows 10 ou superior.
2. **Ferramentas de desenvolvimento:**
   - [Visual Studio 2022](https://visualstudio.microsoft.com/) (versão recente).
   - Workload de **Desenvolvimento para Desktop com .NET** instalado.
3. **SDK .NET:**
   - .NET 8.0 SDK (ou a versão alvo configurada no projeto).
4. **Pacotes NuGet:**
   - `Microsoft.ML.OnnxRuntime`
   - (Opcional) `Microsoft.ML.OnnxRuntime.Gpu` se desejar usar CUDA.

