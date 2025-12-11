using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using System.Drawing.Drawing2D;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace treinamento
{
    public partial class Form1 : Form
    {
        // --- Configuração (Verifique se o caminho do modelo está correto) ---
        private const string ModelPath = @"C:\Users\Natália Marra\source\repos\treinamento\treinamento\best_ir11.onnx";
        private const int InputWidth = 512;
        private const int InputHeight = 512;
        private const float DefaultConfThreshold = 0.25f;
        private const float NmsIouThreshold = 0.45f;

        // Cores e Estilos
        private readonly Color BBoxColor = Color.Red;
        private readonly Color LabelBackColor = Color.Yellow;
        private readonly Color LabelForeColor = Color.Black;
        private readonly int BoxThickness = 3;

        // --- Fields ---
        private InferenceSession? _session;
        private Bitmap? _originalImage;

        public Form1()
        {
            InitializeComponent();
            extBoxConf.Text = DefaultConfThreshold.ToString("F2", CultureInfo.InvariantCulture);

            try
            {
                if (!File.Exists(ModelPath))
                {
                    throw new FileNotFoundException($"O arquivo do modelo não foi encontrado em: {ModelPath}");
                }

                // Opcional: Adicionar SessionOptions para otimizar o uso da CPU
                var options = new SessionOptions();
                // Reduz uso de memória nativa pelo ORT ao desabilitar a arena de CPU
                options.EnableCpuMemArena = false;
                // Habilita otimizações no grafo para melhor desempenho
                options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
                // options.IntraOpNumThreads = Environment.ProcessorCount; // Ajustar threads se necessário

                _session = new InferenceSession(ModelPath, options);
                lblStatus.Text = "Modelo carregado com sucesso.";
            }
            catch (Exception ex)
            {
                lblStatus.Text = "ERRO: Falha ao carregar o modelo.";
                MessageBox.Show($"Erro ao carregar o modelo ONNX: {ex.Message}\n\nVerifique o caminho do arquivo.", "Erro Fatal", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            base.OnFormClosing(e);
            _session?.Dispose();
            _session = null;
        }

        // ----------------------
        // Handlers de Eventos
        // ----------------------

        private void button1_Click(object sender, EventArgs e) // Assumindo 'button1' é o botão de carregar imagem
        {
            using var openFileDialog = new OpenFileDialog
            {
                Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp",
                Title = "Selecione um arquivo de imagem"
            };

            if (openFileDialog.ShowDialog() != DialogResult.OK)
                return;

            try
            {
                // 1. Liberação de recursos antigos (muito importante)
                _originalImage?.Dispose();
                pictureBox1.Image?.Dispose();

                // 2. Carrega a imagem sem bloquear o arquivo
                byte[] imageBytes = File.ReadAllBytes(openFileDialog.FileName);
                using (var stream = new MemoryStream(imageBytes))
                {
                    _originalImage = new Bitmap(stream);
                }

                // 3. Exibição e Clonagem
                // Atribui uma CÓPIA para a PictureBox para que o original possa ser processado
                pictureBox1.Image = (Image)_originalImage.Clone();

                lblStatus.Text = $"Imagem carregada: {Path.GetFileName(openFileDialog.FileName)} ({_originalImage.Width}x{_originalImage.Height})";
                lbResults.Items.Clear();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Erro ao carregar a imagem: {ex.Message}", "Erro", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void Detectar_Click(object sender, EventArgs e) // Assumindo 'Detectar' é o botão de execução
        {
            if (_originalImage == null || _session == null)
            {
                MessageBox.Show(_originalImage == null ? "Nenhuma imagem carregada." : "Modelo não carregado.", "Aviso", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            // 1. Leitura do Limiar de Confiança
            if (!float.TryParse(extBoxConf.Text.Replace(',', '.'), NumberStyles.Float, CultureInfo.InvariantCulture, out float confThreshold))
            {
                confThreshold = DefaultConfThreshold;
                extBoxConf.Text = confThreshold.ToString("F2", CultureInfo.InvariantCulture);
                MessageBox.Show("Valor de confiança inválido. Usando o padrão: " + confThreshold, "Aviso", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }

            try
            {
                // 2. Pré-processamento e Inferência
                var inputTensor = Preprocess(_originalImage);

                // Não descarte imediatamente a imagem exibida; substituiremos após criar a anotada

                // Cria uma cópia da imagem original para desenhar as caixas
                Bitmap originalToDrawOn = (Bitmap)_originalImage.Clone();

                // Executa inferência e pós-processamento
                var predictions = RunInferenceAndDecode(inputTensor, confThreshold);
                var finalPreds = ApplyNms(predictions, NmsIouThreshold);

                // Desenha e atualiza a UI
                var annotated = DrawPredictionsOnOriginal(originalToDrawOn, finalPreds);

                // Substitui a imagem de forma segura e então libera a antiga
                var oldImage = pictureBox1.Image;
                pictureBox1.Image = annotated;
                oldImage?.Dispose();

                lbResults.Items.Clear();
                foreach (var p in finalPreds)
                {
                    lbResults.Items.Add($"Classe {p.ClassId}: {p.Score:P1} | Box: {p.Box.X:F0}, {p.Box.Y:F0}, {p.Box.Width:F0}, {p.Box.Height:F0}");
                }

                lblStatus.Text = $"Detecção concluída. {finalPreds.Count} objetos (conf>{confThreshold:P0})";
            }
            catch (Exception ex)
            {
                lblStatus.Text = $"ERRO durante a inferência: {ex.Message}";
                MessageBox.Show($"Ocorreu um erro durante a detecção: {ex.Message}", "Erro", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        // ----------------------
        // Métodos do Pipeline de Detecção
        // ----------------------

        /// <summary>
        /// Converte a imagem de entrada para tensor float [1,3,H,W], fazendo o redimensionamento interno.
        /// O recurso 'resized' é liberado automaticamente pelo 'using'.
        /// </summary>
        private Tensor<float> Preprocess(Bitmap sourceImage)
        {
            // O 'using' garante que o recurso 'resized' seja liberado
            using Bitmap resized = new Bitmap(InputWidth, InputHeight);
            using (Graphics g = Graphics.FromImage(resized))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.CompositingQuality = CompositingQuality.HighQuality;
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.DrawImage(sourceImage, 0, 0, InputWidth, InputHeight);
            }

            // --- Conversão para Tensor NCHW ---
            int width = resized.Width;
            int height = resized.Height;
            var tensor = new DenseTensor<float>(new[] { 1, 3, height, width });

            // Bloqueia os bits para acesso direto à memória (mais rápido que GetPixel)
            var bitmapData = resized.LockBits(new Rectangle(0, 0, width, height),
                                              System.Drawing.Imaging.ImageLockMode.ReadOnly,
                                              System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            try
            {
                int stride = Math.Abs(bitmapData.Stride);
                int bytes = stride * height;
                byte[] rgbValues = new byte[bytes];
                Marshal.Copy(bitmapData.Scan0, rgbValues, 0, bytes);

                for (int y = 0; y < height; y++)
                {
                    int rowStart = y * stride;
                    for (int x = 0; x < width; x++)
                    {
                        int idx = rowStart + x * 3;

                        // Format24bppRgb é BGR (Blue, Green, Red)
                        byte b = rgbValues[idx + 0];
                        byte g = rgbValues[idx + 1];
                        byte r = rgbValues[idx + 2];

                        // Normalização (dividir por 255) e atribuição NCHW (C, H, W)
                        tensor[0, 0, y, x] = r / 255f; // R
                        tensor[0, 1, y, x] = g / 255f; // G
                        tensor[0, 2, y, x] = b / 255f; // B
                    }
                }
            }
            finally
            {
                resized.UnlockBits(bitmapData);
            }

            return tensor;
        }

        /// <summary>
        /// Executa a inferência e decodifica o output em previsões YOLO.
        /// Garante a liberação dos recursos de inferência.
        /// </summary>
        private List<YoloPrediction> RunInferenceAndDecode(Tensor<float> inputTensor, float confThreshold)
        {
            if (_session == null) return new List<YoloPrediction>();

            var inputName = _session.InputMetadata.Keys.First();
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };

            // O 'using' garante a liberação dos recursos da inferência (IValue tensors)
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
            var outTensor = results.FirstOrDefault()?.AsTensor<float>();
            if (outTensor == null) return new List<YoloPrediction>();

            var dims = outTensor.Dimensions.ToArray();
            var preds = new List<YoloPrediction>();

            // --- Lógica de Detecção do Formato (YOLOv8 vs Outros) ---
            int N_candidates;
            int C_attributes;
            bool isTransposed = false;

            int dim1 = dims[dims.Length - 2];
            int dim2 = dims[dims.Length - 1];

            // Se C (atributos) é menor que N (caixas), a ordem mais provável é [C, N] (transposto)
            if (dim1 < dim2)
            {
                C_attributes = dim1;
                N_candidates = dim2;
                isTransposed = true;
            }
            // Se C (atributos) é maior que N (caixas), a ordem mais provável é [N, C] (normal)
            else
            {
                N_candidates = dim1;
                C_attributes = dim2;
                isTransposed = false;
            }

            // Verificação de segurança: C_attributes deve ser >= 6 (4 caixas + 1 objConf + 1 classe)
            if (C_attributes < 6)
            {
                N_candidates = dim1;
                C_attributes = dim2;
                isTransposed = false;
            }

            int numClasses = C_attributes - 5; // cx, cy, w, h, obj_conf, (classes...)

            // Funções auxiliares para acessar o tensor
            float getFloat(int candidateIndex, int attributeIndex)
            {
                // Para o formato [1, N, C] ou [N, C]:
                if (!isTransposed)
                {
                    // outTensor[Batch=0, N_candidate=candidateIndex, Attribute=attributeIndex]
                    return (dims.Length == 3) ? outTensor[0, candidateIndex, attributeIndex] : outTensor[candidateIndex, attributeIndex];
                }
                // Para o formato [1, C, N] ou [C, N] (transposto):
                else
                {
                    // outTensor[Batch=0, Attribute=attributeIndex, N_candidate=candidateIndex]
                    return (dims.Length == 3) ? outTensor[0, attributeIndex, candidateIndex] : outTensor[attributeIndex, candidateIndex];
                }
            }


            for (int i = 0; i < N_candidates; i++)
            {
                float cx = getFloat(i, 0);
                float cy = getFloat(i, 1);
                float w = getFloat(i, 2);
                float h = getFloat(i, 3);
                float obj = getFloat(i, 4); // Score do objeto (se o modelo o retornar)

                float bestClassProb = 1f;
                int bestClass = 0;

                // Se houver classes (C_attributes > 5)
                if (numClasses > 0)
                {
                    bestClassProb = 0f;
                    bestClass = -1;

                    for (int c = 0; c < numClasses; c++)
                    {
                        float prob = getFloat(i, c + 5);
                        if (prob > bestClassProb)
                        {
                            bestClassProb = prob;
                            bestClass = c;
                        }
                    }
                }

                // Cálculo da pontuação final
                float score = obj * bestClassProb;

                // Filtragem
                if (score < confThreshold) continue;

                // Converte (cx, cy, w, h) para (x_min, y_min, width, height)
                float x = cx - w / 2f;
                float y = cy - h / 2f;

                preds.Add(new YoloPrediction
                {
                    Box = new RectangleF(x, y, w, h),
                    Score = score,
                    ClassId = bestClass
                });
            }

            return preds;
        }

        /// <summary>
        /// Non-maximum suppression (NMS) para remover boxes sobrepostas.
        /// </summary>
        private List<YoloPrediction> ApplyNms(List<YoloPrediction> preds, float iouThreshold)
        {
            var results = new List<YoloPrediction>();
            var byClass = preds.GroupBy(p => p.ClassId);

            foreach (var group in byClass)
            {
                // Ordena por Score descendente
                var list = group.OrderByDescending(p => p.Score).ToList();
                while (list.Count > 0)
                {
                    var best = list[0];
                    results.Add(best);
                    list.RemoveAt(0);

                    // Remove boxes com alto IoU (sobreposição) com a 'best'
                    for (int i = list.Count - 1; i >= 0; i--)
                    {
                        if (IoU(best.Box, list[i].Box) > iouThreshold)
                            list.RemoveAt(i);
                    }
                }
            }
            return results;
        }

        /// <summary>
        /// Calcula a Interseção sobre União (IoU).
        /// </summary>
        private float IoU(RectangleF a, RectangleF b)
        {
            float areaA = a.Width * a.Height;
            float areaB = b.Width * b.Height;

            float interXmin = Math.Max(a.Left, b.Left);
            float interYmin = Math.Max(a.Top, b.Top);
            float interXmax = Math.Min(a.Right, b.Right);
            float interYmax = Math.Min(a.Bottom, b.Bottom);

            float interW = interXmax - interXmin;
            float interH = interYmax - interYmin;

            if (interW <= 0 || interH <= 0) return 0f;

            float interArea = interW * interH;

            // IoU = Area de Interseção / (Area A + Area B - Area de Interseção)
            return interArea / (areaA + areaB - interArea);
        }

        /// <summary>
        /// Desenha previsões na imagem original, mapeando coordenadas do modelo (512x512) para o tamanho original.
        /// Garante a liberação de todos os recursos GDI+ com o 'using'.
        /// </summary>
        private Bitmap DrawPredictionsOnOriginal(Bitmap original, List<YoloPrediction> preds)
        {
            float scaleX = (float)original.Width / InputWidth;
            float scaleY = (float)original.Height / InputHeight;

            // O 'using' garante que o objeto Graphics e todos os recursos de desenho sejam liberados
            using (Graphics g = Graphics.FromImage(original))
            {
                g.SmoothingMode = SmoothingMode.HighQuality;
                using Pen pen = new Pen(BBoxColor, BoxThickness);
                using Font font = new Font("Arial", 12, FontStyle.Bold);
                using SolidBrush backBrush = new SolidBrush(LabelBackColor);
                using SolidBrush textBrush = new SolidBrush(LabelForeColor);

                foreach (var p in preds)
                {
                    // 1. Mapeamento para coordenadas na imagem original
                    float x_original = p.Box.X * scaleX;
                    float y_original = p.Box.Y * scaleY;
                    float w_original = p.Box.Width * scaleX;
                    float h_original = p.Box.Height * scaleY;

                    var rect = new RectangleF(x_original, y_original, w_original, h_original);
                    g.DrawRectangle(pen, rect.X, rect.Y, rect.Width, rect.Height);

                    // 2. Desenha o Label
                    string label = $"Classe {p.ClassId}: {p.Score:P1}";
                    var textSize = g.MeasureString(label, font);

                    float labelY = Math.Max(0, rect.Y - textSize.Height);
                    var labelRect = new RectangleF(rect.X, labelY, textSize.Width + 5, textSize.Height);

                    g.FillRectangle(backBrush, labelRect);
                    g.DrawString(label, font, textBrush, labelRect.Location);
                }
            }
            // A imagem original/clonada (agora anotada) é retornada.
            return original;
        }
    }

    /// <summary>
    /// Classe de dados para armazenar a predição de um objeto YOLO.
    /// </summary>
    public class YoloPrediction
    {
        // Coordenadas normalizadas pelo tamanho do modelo (ex: 512x512)
        public RectangleF Box { get; set; }
        public float Score { get; set; }
        public int ClassId { get; set; }
    }
}