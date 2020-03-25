using System;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Structure;
using System.Collections.Generic;
using System.Drawing;
using Emgu.CV.Dnn;
namespace CudaUI
{
    public partial class MainForm : Form
    {
      
        public MainForm()
        {
            InitializeComponent();
        }
        
        bool blCameraOpen = false;
        VideoCapture objVideoCapture;

        private void button1_Click(object sender, EventArgs e)
        {
            blCameraOpen = !blCameraOpen;

            if (blCameraOpen)
            {
                // 啟動照相機
                btnStartCamera.Text = "Stop";
                objVideoCapture.Start();
                timer1.Enabled = true;
            }
            else
            {
                // 停止照相機
                btnStartCamera.Text = "Start";
                objVideoCapture.Stop();
                timer1.Enabled = false;
            }
        }

        private void MainForm_Load(object sender, EventArgs e)
        {
            timer1.Enabled = false;
            objVideoCapture = new VideoCapture(0);
            objVideoCapture.ImageGrabbed += ProcessFrameAsync;
        }
        Mat RunImg = new Mat();
        private void ProcessFrameAsync(object sender, EventArgs e)
        {
            try
            {
                using (Mat objMat = new Mat())
                {
                    objVideoCapture.Retrieve(objMat);
                    RunImg = new Mat();
                    objMat.CopyTo(RunImg);                    


                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }
        
        public static void Detect(Mat image, List<Rectangle> faces)
        {

            Net net = DnnInvoke.ReadNetFromTensorflow(@"C:\models\opencv_face_detector_uint8.pb", @"C:\models\opencv_face_detector.pbtxt");
            Mat inputBlob = DnnInvoke.BlobFromImage(image, 1.0, new Size(300, 300), new MCvScalar(104.0, 117.0, 123.0), true, false);
            net.SetInput(inputBlob, "data");
            Mat detection = net.Forward("detection_out");

            int resultRows = detection.SizeOfDimension[2];
            int resultCols = detection.SizeOfDimension[3];

            float[] temp = new float[resultRows * resultCols];
            Marshal.Copy(detection.DataPointer, temp, 0, temp.Length);

            for (int i = 0; i < resultRows; i++)
            {
                float confidence = temp[i * resultCols + 2];
                if (confidence > 0.7)
                {
                    int x1 = (int)(temp[i * resultCols + 3] * image.Width);
                    int y1 = (int)(temp[i * resultCols + 4] * image.Height);
                    int x2 = (int)(temp[i * resultCols + 5] * image.Width);
                    int y2 = (int)(temp[i * resultCols + 6] * image.Height);

                    Rectangle rectangle = new Rectangle(x1, y1, x2 - x1, y2 - y1);
                    faces.Add(rectangle);
                }
            }
        }
        Mat tmp = new Mat();
        private void timer1_Tick(object sender, EventArgs e)
        {
            timer1.Enabled = false;
            if (!RunImg.IsEmpty)
            {
                tmp = RunImg;
                List<Rectangle> faces = new List<Rectangle>();
                Detect(tmp, faces);
                for (int f = 0; f < faces.Count; f++)
                    CvInvoke.Rectangle(tmp, faces[f], new Bgr(Color.Red).MCvScalar, 2);
                imageBox1.Image = tmp;
            }
            timer1.Enabled = true;
        }
    }
}
