using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace treinamento
{
    public partial class Form1 : Form
    {
        // --- Configuração (ajustar caminho do modelo) ---
        private const string ModelPath = @"C:\Users\epuhl\source\repos\Deteccao-onnx\treinamento\best_ir11.onnx";
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
                    throw new FileNotFoundException($"O arquivo do modelo não foi encontrado em: {ModelPath}");

                // IMPORTANTE: no projeto, configurar PlatformTarget = x64

                var options = new SessionOptions
                {
                    EnableCpuMemArena = false,
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
                };

                _session = new InferenceSession(ModelPath, options);
                lblStatus.Text = "Modelo carregado com sucesso.";
            }
            catch (Exception ex)
            {
                lblStatus.Text = "ERRO: Falha ao carregar o modelo.";
                MessageBox.Show(
                    $"Erro ao carregar o modelo ONNX:\n\n{ex}",
                    "Erro Fatal",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
            }
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            base.OnFormClosing(e);
            _session?.Dispose();
            _session = null;
        }

        // =========================================================
        // EVENTOS
        // =========================================================

        // Botão "Carregar imagem"
        private void button1_Click(object sender, EventArgs e)
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
                // Libera recursos antigos
                _originalImage?.Dispose();
                _originalImage = null;

                if (pictureBox1.Image != null)
                {
                    pictureBox1.Image.Dispose();
                    pictureBox1.Image = null;
                }

                // Carrega sem travar o arquivo
                byte[] imageBytes = File.ReadAllBytes(openFileDialog.FileName);
                using (var stream = new MemoryStream(imageBytes))
                {
                    _originalImage = new Bitmap(stream);
                }

                // Mostra cópia na PictureBox
                pictureBox1.Image = (Image)_originalImage.Clone();

                lblStatus.Text = $"Imagem carregada: {Path.GetFileName(openFileDialog.FileName)} ({_originalImage.Width}x{_originalImage.Height})";
                lbResults.Items.Clear();
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"Erro ao carregar a imagem:\n\n{ex}",
                    "Erro",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
            }
        }

        // Botão "Detectar"
        private void Detectar_Click(object sender, EventArgs e)
        {
            if (_originalImage == null || _session == null)
            {
                MessageBox.Show(
                    _originalImage == null ? "Nenhuma imagem carregada." : "Modelo não carregado.",
                    "Aviso",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning);
                return;
            }

            // Lê limiar de confiança
            if (!float.TryParse(extBoxConf.Text.Replace(',', '.'),
                                NumberStyles.Float,
                                CultureInfo.InvariantCulture,
                                out float confThreshold))
            {
                confThreshold = DefaultConfThreshold;
                extBoxConf.Text = confThreshold.ToString("F2", CultureInfo.InvariantCulture);
                MessageBox.Show(
                    "Valor de confiança inválido. Usando o padrão: " + confThreshold,
                    "Aviso",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning);
            }

            try
            {
                // 1) Pré-processamento
                var inputTensor = Preprocess(_originalImage);

                // 2) Inferência + pós-processamento
                var predictions = RunInferenceAndDecode(inputTensor, confThreshold);
                var finalPreds = ApplyNms(predictions, NmsIouThreshold);

                // 3) Desenho em cópia não-indexada da imagem original
                Bitmap annotated = DrawPredictionsOnOriginal(_originalImage, finalPreds);

                // 4) Atualiza UI
                var oldImage = pictureBox1.Image;
                pictureBox1.Image = annotated;
                oldImage?.Dispose();

                lbResults.Items.Clear();
                foreach (var p in finalPreds)
                {
                    lbResults.Items.Add(
                        $"Classe {p.ClassId}: {p.Score:P1} | Box: {p.Box.X:F0}, {p.Box.Y:F0}, {p.Box.Width:F0}, {p.Box.Height:F0}");
                }

                lblStatus.Text = $"Detecção concluída. {finalPreds.Count} objetos (conf>{confThreshold:P0})";
            }
            catch (Exception ex)
            {
                lblStatus.Text = $"ERRO durante a inferência.";
                MessageBox.Show(
                    $"Ocorreu um erro durante a detecção:\n\n{ex}",
                    "Erro",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {
        }

        // =========================================================
        // PIPELINE: PREPROCESS, INFER, NMS, DRAW
        // =========================================================

        // Bitmap -> Tensor [1,3,H,W]
        private Tensor<float> Preprocess(Bitmap sourceImage)
        {
            using Bitmap resized = new Bitmap(InputWidth, InputHeight);
            using (Graphics g = Graphics.FromImage(resized))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.CompositingQuality = CompositingQuality.HighQuality;
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.DrawImage(sourceImage, 0, 0, InputWidth, InputHeight);
            }

            int width = resized.Width;
            int height = resized.Height;
            var tensor = new DenseTensor<float>(new[] { 1, 3, height, width });

            var bitmapData = resized.LockBits(
                new Rectangle(0, 0, width, height),
                ImageLockMode.ReadOnly,
                PixelFormat.Format24bppRgb);

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

                        byte b = rgbValues[idx + 0];
                        byte g = rgbValues[idx + 1];
                        byte r = rgbValues[idx + 2];

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

        // INFERÊNCIA + decodificação YOLOv8/YOLO11
        private List<YoloPrediction> RunInferenceAndDecode(Tensor<float> inputTensor, float confThreshold)
        {
            if (_session == null) return new List<YoloPrediction>();

            var inputName = _session.InputMetadata.Keys.First();

            // DESCARTA corretamente o NamedOnnxValue
             var inputOnnxValue = NamedOnnxValue.CreateFromTensor(inputName, inputTensor);

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results =
                _session.Run(new[] { inputOnnxValue });

            var outTensor = results.FirstOrDefault()?.AsTensor<float>();
            if (outTensor == null) return new List<YoloPrediction>();

            var dims = outTensor.Dimensions.ToArray();  // ex: [1, 84, 8400] ou [1, 8400, 84]

            int dim1 = dims[dims.Length - 2];
            int dim2 = dims[dims.Length - 1];

            int C_attributes;   // 4 box + numClasses
            int N_candidates;
            bool isTransposed;

            if (dim1 < dim2)
            {
                // [1, C, N]
                C_attributes = dim1;
                N_candidates = dim2;
                isTransposed = true;
            }
            else
            {
                // [1, N, C]
                N_candidates = dim1;
                C_attributes = dim2;
                isTransposed = false;
            }

            // YOLOv8/11: [cx, cy, w, h, class0, class1, ...]
            int numClasses = C_attributes - 4;
            if (numClasses <= 0) numClasses = 1;

            float getFloat(int candidateIndex, int attributeIndex)
            {
                if (!isTransposed)
                {
                    // [1, N, C] ou [N, C]
                    return (dims.Length == 3)
                        ? outTensor[0, candidateIndex, attributeIndex]
                        : outTensor[candidateIndex, attributeIndex];
                }
                else
                {
                    // [1, C, N] ou [C, N]
                    return (dims.Length == 3)
                        ? outTensor[0, attributeIndex, candidateIndex]
                        : outTensor[attributeIndex, candidateIndex];
                }
            }

            var preds = new List<YoloPrediction>();
            float maxScore = 0f;

            for (int i = 0; i < N_candidates; i++)
            {
                // cx, cy, w, h
                float cx = getFloat(i, 0);
                float cy = getFloat(i, 1);
                float w = getFloat(i, 2);
                float h = getFloat(i, 3);

                // Classes começam em 4
                float bestClassProb = 0f;
                int bestClass = -1;
                for (int c = 0; c < numClasses; c++)
                {
                    float prob = getFloat(i, 4 + c);
                    if (prob > bestClassProb)
                    {
                        bestClassProb = prob;
                        bestClass = c;
                    }
                }

                float score = bestClassProb;   // sem obj_conf

                if (score < confThreshold)
                    continue;

                if (score > maxScore) maxScore = score;

                float x = cx - w / 2f;
                float y = cy - h / 2f;

                preds.Add(new YoloPrediction
                {
                    Box = new RectangleF(x, y, w, h),
                    Score = score,
                    ClassId = bestClass
                });
            }

#if DEBUG
            // Debug: ver se o modelo está respondendo
            MessageBox.Show(
                $"ONNX executado.\n" +
                $"Saída: [{string.Join(" x ", dims)}]\n" +
                $"Candidatos (N): {N_candidates}\n" +
                $"Atributos (C): {C_attributes}\n" +
                $"Classes: {numClasses}\n" +
                $"Maior score bruto: {maxScore:F3}",
                "Debug YOLO / ONNX");
#endif

            return preds;
        }

        // NMS
        private List<YoloPrediction> ApplyNms(List<YoloPrediction> preds, float iouThreshold)
        {
            var results = new List<YoloPrediction>();
            var byClass = preds.GroupBy(p => p.ClassId);

            foreach (var group in byClass)
            {
                var list = group.OrderByDescending(p => p.Score).ToList();
                while (list.Count > 0)
                {
                    var best = list[0];
                    results.Add(best);
                    list.RemoveAt(0);

                    for (int i = list.Count - 1; i >= 0; i--)
                    {
                        if (IoU(best.Box, list[i].Box) > iouThreshold)
                            list.RemoveAt(i);
                    }
                }
            }
            return results;
        }

        // IoU
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

            return interArea / (areaA + areaB - interArea);
        }

        // Desenha previsões numa cópia 24bpp da imagem original
        private Bitmap DrawPredictionsOnOriginal(Bitmap original, List<YoloPrediction> preds)
        {
            // Cria bitmap não indexado para desenhar
            Bitmap annotated = new Bitmap(
                original.Width,
                original.Height,
                PixelFormat.Format24bppRgb);

            float scaleX = (float)original.Width / InputWidth;
            float scaleY = (float)original.Height / InputHeight;

            using (Graphics g = Graphics.FromImage(annotated))
            {
                // Primeiro desenha a imagem original
                g.SmoothingMode = SmoothingMode.HighQuality;
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.DrawImage(original, 0, 0, original.Width, original.Height);

                using Pen pen = new Pen(BBoxColor, BoxThickness);
                using Font font = new Font("Arial", 12, FontStyle.Bold);
                using SolidBrush backBrush = new SolidBrush(LabelBackColor);
                using SolidBrush textBrush = new SolidBrush(LabelForeColor);

                foreach (var p in preds)
                {
                    float x_original = p.Box.X * scaleX;
                    float y_original = p.Box.Y * scaleY;
                    float w_original = p.Box.Width * scaleX;
                    float h_original = p.Box.Height * scaleY;

                    var rect = new RectangleF(x_original, y_original, w_original, h_original);
                    g.DrawRectangle(pen, rect.X, rect.Y, rect.Width, rect.Height);

                    string label = $"Classe {p.ClassId}: {p.Score:P1}";
                    var textSize = g.MeasureString(label, font);

                    float labelY = Math.Max(0, rect.Y - textSize.Height);
                    var labelRect = new RectangleF(rect.X, labelY, textSize.Width + 5, textSize.Height);

                    g.FillRectangle(backBrush, labelRect);
                    g.DrawString(label, font, textBrush, labelRect.Location);
                }
            }

            return annotated;
        }
    }

    // Classe de predição
    public class YoloPrediction
    {
        public RectangleF Box { get; set; }
        public float Score { get; set; }
        public int ClassId { get; set; }
    }
}

