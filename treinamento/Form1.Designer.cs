namespace treinamento
{
    partial class Form1
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            pictureBox1 = new PictureBox();
            btnLoadImage = new Button();
            btnDetect = new Button();
            extBoxConf = new TextBox();
            lbResults = new ListBox();
            lblStatus = new Label();
            ((System.ComponentModel.ISupportInitialize)pictureBox1).BeginInit();
            SuspendLayout();
            // 
            // pictureBox1
            // 
            pictureBox1.Location = new Point(12, 8);
            pictureBox1.Name = "pictureBox1";
            pictureBox1.Size = new Size(624, 430);
            pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox1.TabIndex = 0;
            pictureBox1.TabStop = false;
            // 
            // btnLoadImage
            // 
            btnLoadImage.Location = new Point(695, 77);
            btnLoadImage.Name = "btnLoadImage";
            btnLoadImage.Size = new Size(129, 51);
            btnLoadImage.TabIndex = 1;
            btnLoadImage.Text = "Carregar imagem";
            btnLoadImage.UseVisualStyleBackColor = true;
            btnLoadImage.Click += button1_Click;
            // 
            // btnDetect
            // 
            btnDetect.Location = new Point(695, 134);
            btnDetect.Name = "btnDetect";
            btnDetect.Size = new Size(129, 66);
            btnDetect.TabIndex = 2;
            btnDetect.Text = "Detectar";
            btnDetect.UseVisualStyleBackColor = true;
            btnDetect.Click += Detectar_Click;
            // 
            // extBoxConf
            // 
            extBoxConf.Location = new Point(695, 44);
            extBoxConf.Name = "extBoxConf";
            extBoxConf.Size = new Size(120, 27);
            extBoxConf.TabIndex = 3;
            // 
            // lbResults
            // 
            lbResults.FormattingEnabled = true;
            lbResults.Location = new Point(642, 216);
            lbResults.Name = "lbResults";
            lbResults.Size = new Size(221, 84);
            lbResults.TabIndex = 4;
            // 
            // lblStatus
            // 
            lblStatus.AutoSize = true;
            lblStatus.Location = new Point(25, 408);
            lblStatus.Name = "lblStatus";
            lblStatus.Size = new Size(53, 20);
            lblStatus.TabIndex = 5;
            lblStatus.Text = "Pronto";
            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(8F, 20F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(875, 450);
            Controls.Add(lblStatus);
            Controls.Add(lbResults);
            Controls.Add(extBoxConf);
            Controls.Add(btnDetect);
            Controls.Add(btnLoadImage);
            Controls.Add(pictureBox1);
            Name = "Form1";
            Text = "Form1";
            Load += Form1_Load;
            ((System.ComponentModel.ISupportInitialize)pictureBox1).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private PictureBox pictureBox1;
        private Button btnLoadImage;
        private Button btnDetect;
        private TextBox extBoxConf;
        private ListBox lbResults;
        private Label lblStatus;
    }
}
